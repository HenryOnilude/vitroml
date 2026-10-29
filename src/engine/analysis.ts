import { forward, createRunBuffers } from './forward';
import type { ModelConfig, ModelWeights } from './model';
import { HookRegistry } from './hooks';
import { zeroAblateHead } from './interventions';
import type { Tokenizer } from './tokenizer';
import { Tensor } from './tensor';
import { rmsnorm } from './ops/rmsnorm';
import { matmul } from './ops/matmul';

export interface TokenPrediction {
  tokenId: number;
  tokenPiece: string;
  logit: number;
  delta?: number;
}

export interface AblationComparisonResult {
  targetLayer: number;
  targetHead: number;
  testedPos: number;
  testedToken: number;
  testedPiece: string;
  baselineTop5: TokenPrediction[];
  ablatedTop5: TokenPrediction[];
  l2Diff: number;
  maxAbsDiff: number;
  predictionFlipped: boolean;
  baselineTop: TokenPrediction;
  ablatedTop: TokenPrediction;
}

/**
 * Logit Lens projection: Applies final RMSNorm and unembedding classifier
 * to an intermediate residual stream tensor to preview top predictions mid-run.
 */
export function computeLogitLens(
  residualTensor: Tensor,
  config: ModelConfig,
  weights: ModelWeights,
  tokenizer: Tokenizer,
  prevToken: number,
  topK: number = 3
): TokenPrediction[] {
  const normTensor = new Tensor(new Float32Array(config.dim), [config.dim]);
  const logitsTensor = new Tensor(new Float32Array(config.vocabSize), [config.vocabSize]);

  // 1. RMSNorm on intermediate residual
  rmsnorm(
    normTensor,
    residualTensor,
    weights.rmsFinalWeight,
    config.dim
  );

  // 2. Project through shared unembedding classifier W_U (wcls)
  matmul(
    logitsTensor,
    normTensor,
    weights.wcls,
    config.dim,
    config.vocabSize
  );

  // 3. Extract top-K
  const logitsData = logitsTensor.data;
  const logitsOff = logitsTensor.offset;

  const ranked: { idx: number; logit: number }[] = [];
  for (let i = 0; i < config.vocabSize; i++) {
    ranked.push({ idx: i, logit: logitsData[logitsOff + i] });
  }

  ranked.sort((a, b) => b.logit - a.logit);
  const topTokens = ranked.slice(0, topK);

  return topTokens.map((item) => ({
    tokenId: item.idx,
    tokenPiece: tokenizer.decode(prevToken, item.idx),
    logit: item.logit,
  }));
}

/**
 * Greedy decoding: returns the vocabulary index with the highest logit.
 */
export function argmaxToken(logits: Tensor): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.size; i++) {
    const v = logits.data[logits.offset + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Runs one clean forward pass and returns the raw logit vector.
 * Used as the baseline for per-head ablation importance scans.
 */
export function probeBaselineLogits(
  inputToken: number,
  pos: number,
  config: ModelConfig,
  weights: ModelWeights
): Float32Array {
  const t = forward(inputToken, pos, config, weights, createRunBuffers(config), new HookRegistry());
  return Float32Array.from(t.data.subarray(t.offset, t.offset + t.size));
}

/**
 * Measures how much a single head matters: zero-ablates it, re-runs the forward
 * pass, and returns the L2 distance between ablated and baseline logits.
 */
export function measureHeadAblationL2(
  baselineLogits: Float32Array,
  inputToken: number,
  pos: number,
  layer: number,
  head: number,
  config: ModelConfig,
  weights: ModelWeights
): number {
  const bufs = createRunBuffers(config);
  const hooks = new HookRegistry();
  zeroAblateHead(hooks, layer, head, config.nHeads, config.dim);
  const t = forward(inputToken, pos, config, weights, bufs, hooks);
  let ss = 0;
  for (let i = 0; i < config.vocabSize; i++) {
    const d = t.data[t.offset + i] - baselineLogits[i];
    ss += d * d;
  }
  return Math.sqrt(ss);
}

/**
 * Runs a baseline forward pass and an intervened (zero-ablated) forward pass,
 * comparing the resulting output logits and top-5 token predictions.
 * Accepts either a single token or a sequence of tokens to populate KV cache up to pos.
 */
export function compareHeadAblation(
  tokensOrToken: number[] | number,
  pos: number,
  targetLayer: number,
  targetHead: number,
  config: ModelConfig,
  weights: ModelWeights,
  tokenizer: Tokenizer
): AblationComparisonResult {
  const tokenList = Array.isArray(tokensOrToken) ? tokensOrToken : [tokensOrToken];
  const maxPos = Math.min(pos, tokenList.length - 1);
  const currentToken = tokenList[maxPos];
  const prevToken = maxPos > 0 ? tokenList[maxPos - 1] : 1;
  const testedPiece = tokenizer.decode(prevToken, currentToken);

  // 1. Baseline Run: populate KV cache up to maxPos
  const baselineBuffers = createRunBuffers(config);
  const baselineHooks = new HookRegistry();
  let baselineLogitsTensor!: Tensor;
  for (let p = 0; p <= maxPos; p++) {
    baselineLogitsTensor = forward(tokenList[p], p, config, weights, baselineBuffers, baselineHooks);
  }
  const baselineLogits = Array.from(
    baselineLogitsTensor.data.subarray(
      baselineLogitsTensor.offset,
      baselineLogitsTensor.offset + config.vocabSize
    )
  );

  // 2. Intervened Run: populate ablated KV cache up to maxPos with target head zeroed
  const ablatedBuffers = createRunBuffers(config);
  const ablatedHooks = new HookRegistry();
  zeroAblateHead(ablatedHooks, targetLayer, targetHead, config.nHeads, config.dim);
  let ablatedLogitsTensor!: Tensor;
  for (let p = 0; p <= maxPos; p++) {
    ablatedLogitsTensor = forward(tokenList[p], p, config, weights, ablatedBuffers, ablatedHooks);
  }
  const ablatedLogits = Array.from(
    ablatedLogitsTensor.data.subarray(
      ablatedLogitsTensor.offset,
      ablatedLogitsTensor.offset + config.vocabSize
    )
  );

  // 3. Compute Metrics
  let maxAbsDiff = 0.0;
  let sumSquaredDiff = 0.0;
  for (let i = 0; i < config.vocabSize; i++) {
    const diff = ablatedLogits[i] - baselineLogits[i];
    const absDiff = Math.abs(diff);
    if (absDiff > maxAbsDiff) {
      maxAbsDiff = absDiff;
    }
    sumSquaredDiff += diff * diff;
  }
  const l2Diff = Math.sqrt(sumSquaredDiff);

  // Top 5 Baseline
  const baselineRanked = baselineLogits
    .map((logit, idx) => ({ idx, logit }))
    .sort((a, b) => b.logit - a.logit)
    .slice(0, 5);

  const baselineTop5: TokenPrediction[] = baselineRanked.map((item) => ({
    tokenId: item.idx,
    tokenPiece: tokenizer.decode(currentToken, item.idx),
    logit: item.logit,
  }));

  // Top 5 Ablated (with deltas against baseline)
  const ablatedRanked = ablatedLogits
    .map((logit, idx) => ({ idx, logit }))
    .sort((a, b) => b.logit - a.logit)
    .slice(0, 5);

  const ablatedTop5: TokenPrediction[] = ablatedRanked.map((item) => ({
    tokenId: item.idx,
    tokenPiece: tokenizer.decode(currentToken, item.idx),
    logit: item.logit,
    delta: item.logit - baselineLogits[item.idx],
  }));

  const baselineTop = baselineTop5[0];
  const ablatedTop = ablatedTop5[0];
  const predictionFlipped = baselineTop.tokenId !== ablatedTop.tokenId;

  return {
    targetLayer,
    targetHead,
    testedPos: maxPos,
    testedToken: currentToken,
    testedPiece,
    baselineTop5,
    ablatedTop5,
    l2Diff,
    maxAbsDiff,
    predictionFlipped,
    baselineTop,
    ablatedTop,
  };
}
