import { forward, createRunBuffers, type StepWrite } from './forward';
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
  /** softmax probability of the baseline top token under each run */
  baselineTopProb: number;
  ablatedTopProb: number;
  /** baseline top-3 tokens with softmax probability under each run (paired bars) */
  top3Pairs: { tokenId: number; tokenPiece: string; baselineProb: number; ablatedProb: number }[];
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

export interface WriteAttribution {
  /** 'head' for attention heads, 'ffn' for the feed-forward block,
   *  'rescale' for the RMS-norm rescale term (non-head contributor) */
  kind: 'head' | 'ffn' | 'rescale';
  layer: number;
  head: number;      // -1 for ffn and rescale
  /** signed change to the compared logit quantity (logitA − logitB) */
  effect: number;
}

/**
 * Exact logit-space attribution of one step's residual write, measured on the
 * difference logit(tokenA) − logit(tokenB).
 *
 * With g = rmsFinalWeight ⊙ (wCls[A] − wCls[B]) and s(x) = sqrt(mean(x²) + eps),
 * the lens quantity is L(x) = (x·g)/s(x). Since x_post = r_before + Σvecs,
 *
 *   L(x_post) − L(r_before)
 *     = Σ_i (vec_i·g)/s_after  +  (r_before·g)(1/s_after − 1/s_before)
 *
 * The first term is decomposed per contributor (head or ffn); the second is
 * returned as its own non-head contributor { kind: 'rescale' }, so the returned
 * effects sum to the exact change in the logit difference (float reorder only).
 */
export function attributeStepWrites(
  write: StepWrite,
  residualPost: Tensor,
  tokenA: number,
  tokenB: number,
  weights: ModelWeights,
  dim: number
): WriteAttribution[] {
  const wCls = weights.wcls.data;
  const clsRowA = weights.wcls.offset + tokenA * dim;
  const clsRowB = weights.wcls.offset + tokenB * dim;
  const wNorm = weights.rmsFinalWeight.data;
  const wNormOff = weights.rmsFinalWeight.offset;

  // g = rmsFinalWeight ⊙ (unembedding vector for the comparison A − B)
  const g = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    g[d] = wNorm[wNormOff + d] * (wCls[clsRowA + d] - wCls[clsRowB + d]);
  }

  const xData = residualPost.data;
  const xOff = residualPost.offset;

  // r_before = x_post − Σ vecs (residual-space closure of the write)
  const rBefore = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    let sum = xData[xOff + d];
    for (const vec of write.vecs) sum -= vec[d];
    rBefore[d] = sum;
  }

  const eps = 1e-5;
  let ssPost = 0;
  let ssBefore = 0;
  for (let d = 0; d < dim; d++) {
    ssPost += xData[xOff + d] * xData[xOff + d];
    ssBefore += rBefore[d] * rBefore[d];
  }
  const invAfter = 1 / Math.sqrt(ssPost / dim + eps);
  const invBefore = 1 / Math.sqrt(ssBefore / dim + eps);

  const attrs: WriteAttribution[] = write.vecs.map((vec, i) => {
    let effect = 0;
    for (let d = 0; d < dim; d++) {
      effect += vec[d] * g[d];
    }
    return {
      kind: write.kind === 'heads' ? 'head' : 'ffn',
      layer: write.layer,
      head: write.kind === 'heads' ? i : -1,
      effect: effect * invAfter,
    };
  });

  // Norm-rescale term: the step also re-scales the prior residual because the
  // new RMS scale differs from the old one. Its own non-head contributor.
  let rBeforeDotG = 0;
  for (let d = 0; d < dim; d++) {
    rBeforeDotG += rBefore[d] * g[d];
  }
  attrs.push({
    kind: 'rescale',
    layer: write.layer,
    head: -1,
    effect: rBeforeDotG * (invAfter - invBefore),
  });

  return attrs;
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

  // Softmax probabilities of the baseline top token and top-3 pairs for the
  // plain-English result view (confidence drop + before/after bars).
  const probOf = (logits: number[], id: number): number => {
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let sum = 0;
    for (const v of logits) sum += Math.exp(v - max);
    return Math.exp(logits[id] - max) / sum;
  };
  const baselineTopProb = probOf(baselineLogits, baselineTop.tokenId);
  const ablatedTopProb = probOf(ablatedLogits, baselineTop.tokenId);
  const top3Pairs = baselineTop5.slice(0, 3).map((t) => ({
    tokenId: t.tokenId,
    tokenPiece: t.tokenPiece,
    baselineProb: probOf(baselineLogits, t.tokenId),
    ablatedProb: probOf(ablatedLogits, t.tokenId),
  }));

  return {
    targetLayer,
    targetHead,
    testedPos: maxPos,
    testedToken: currentToken,
    testedPiece,
    baselineTop5,
    ablatedTop5,
    baselineTopProb,
    ablatedTopProb,
    top3Pairs,
    l2Diff,
    maxAbsDiff,
    predictionFlipped,
    baselineTop,
    ablatedTop,
  };
}
