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
  prob?: number;
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
  /** pre-softmax logit of the clean run's top-1 token, in each run */
  baselineTopLogit: number;
  ablatedTopTokenLogit: number;
  /** baselineTopLogit − ablatedTopTokenLogit. Positive = zeroing this head
   *  weakened the top guess; negative = the guess got stronger. */
  top1LogitDrop: number;
  /** logit margin top1 − top2 in each run */
  baselineMargin: number;
  ablatedMargin: number;
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

  // 3. Extract top-K and compute softmax probabilities
  const logitsData = logitsTensor.data;
  const logitsOff = logitsTensor.offset;

  let maxLogit = -Infinity;
  for (let i = 0; i < config.vocabSize; i++) {
    const l = logitsData[logitsOff + i];
    if (l > maxLogit) maxLogit = l;
  }
  let sumExp = 0;
  for (let i = 0; i < config.vocabSize; i++) {
    sumExp += Math.exp(logitsData[logitsOff + i] - maxLogit);
  }

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
    prob: sumExp > 0 ? Math.exp(item.logit - maxLogit) / sumExp : undefined,
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
 * Per-head impact used for ranking on the circuit map: the drop in the clean
 * run's top-1 token pre-softmax logit when the head is zero-ablated.
 * Positive = the head supported the top guess; negative = zeroing it
 * strengthened the top guess (a suppressor head).
 */
export interface HeadImpact {
  layer: number;
  head: number;
  top1LogitDrop: number;
  /** true when ablation changes the argmax token */
  predictionFlipped: boolean;
  /** top1 − top2 logit margin in the ablated run (baseline margin is scan-level) */
  ablatedMargin: number;
}

export interface HeadImpactScan {
  impacts: (HeadImpact | null)[][];
  baselineLogits: Float32Array;
  baselineTopTokenId: number;
  baselineTopPiece: string;
  /** top1 − top2 logit margin in the clean run */
  baselineMargin: number;
  /** false when stopped early via isCancelled */
  completed: boolean;
}

export interface ScanOptions {
  /** called as each head's measurement lands, for progressive grid fill */
  onCell?: (impact: HeadImpact) => void;
  /** called once the shared clean-run logits are known */
  onBaseline?: (baselineLogits: Float32Array) => void;
  /** checked between token forwards and between heads; scan stops when true */
  isCancelled?: () => boolean;
}

/** Cache key for a (prompt, position) measurement: position + token prefix. */
export function comparisonCacheKey(tokenList: number[], pos: number): string {
  const maxPos = Math.min(pos, tokenList.length - 1);
  return `${maxPos}|${tokenList.slice(0, maxPos + 1).join(',')}`;
}

/** Indices of the two highest logits. */
function top2Indices(logits: ArrayLike<number>): { i1: number; i2: number } {
  let i1 = 0;
  let i2 = 1;
  if (logits[i2] > logits[i1]) {
    i1 = 1;
    i2 = 0;
  }
  for (let i = 2; i < logits.length; i++) {
    const v = logits[i];
    if (v > logits[i1]) {
      i2 = i1;
      i1 = i;
    } else if (v > logits[i2]) {
      i2 = i;
    }
  }
  return { i1, i2 };
}

// Long synchronous chunks yield to the event loop at most every YIELD_BUDGET_MS
// so scans and comparisons never freeze the UI.
const YIELD_BUDGET_MS = 24;

/**
 * Async version of the per-token forward loop used by scans and comparisons.
 * Replays the sequence through the KV cache, yielding to the event loop under
 * a time budget. Returns null when cancelled.
 */
async function runSequenceLogits(
  tokenList: number[],
  maxPos: number,
  hooks: HookRegistry,
  config: ModelConfig,
  weights: ModelWeights,
  isCancelled?: () => boolean
): Promise<Float32Array | null> {
  const bufs = createRunBuffers(config);
  let logitsTensor: Tensor | null = null;
  let budgetStart = performance.now();
  for (let p = 0; p <= maxPos; p++) {
    logitsTensor = forward(tokenList[p], p, config, weights, bufs, hooks);
    if (isCancelled?.()) return null;
    if (performance.now() - budgetStart > YIELD_BUDGET_MS) {
      await new Promise((r) => setTimeout(r, 0));
      budgetStart = performance.now();
    }
  }
  const t = logitsTensor!;
  return Float32Array.from(t.data.subarray(t.offset, t.offset + config.vocabSize));
}

/**
 * Measures every head's top-1 logit drop at a position: one shared clean run,
 * then one zero-ablated run per head, computed asynchronously and reported
 * cell-by-cell via onCell. Returns partial results with completed=false when
 * cancelled — callers should not cache incomplete scans.
 */
export async function scanHeadImpacts(
  tokens: number[],
  pos: number,
  config: ModelConfig,
  weights: ModelWeights,
  tokenizer: Tokenizer,
  opts: ScanOptions = {}
): Promise<HeadImpactScan | null> {
  const maxPos = Math.min(pos, tokens.length - 1);
  const tokenList = tokens.slice(0, maxPos + 1);

  const baselineLogits = await runSequenceLogits(
    tokenList,
    maxPos,
    new HookRegistry(),
    config,
    weights,
    opts.isCancelled
  );
  if (!baselineLogits) return null;
  opts.onBaseline?.(baselineLogits);

  const { i1: top1, i2: top2 } = top2Indices(baselineLogits);
  const baselineMargin = baselineLogits[top1] - baselineLogits[top2];
  const baselineTopPiece = tokenizer.decode(tokenList[maxPos], top1);

  const impacts: (HeadImpact | null)[][] = Array.from({ length: config.nLayers }, () =>
    Array(config.nHeads).fill(null)
  );
  const partial = (): HeadImpactScan => ({
    impacts,
    baselineLogits,
    baselineTopTokenId: top1,
    baselineTopPiece,
    baselineMargin,
    completed: false,
  });

  for (let l = 0; l < config.nLayers; l++) {
    for (let h = 0; h < config.nHeads; h++) {
      if (opts.isCancelled?.()) return partial();
      const hooks = new HookRegistry();
      zeroAblateHead(hooks, l, h, config.nHeads, config.dim);
      const ablated = await runSequenceLogits(
        tokenList,
        maxPos,
        hooks,
        config,
        weights,
        opts.isCancelled
      );
      if (!ablated) return partial();
      const at = top2Indices(ablated);
      const impact: HeadImpact = {
        layer: l,
        head: h,
        top1LogitDrop: baselineLogits[top1] - ablated[top1],
        predictionFlipped: at.i1 !== top1,
        ablatedMargin: ablated[at.i1] - ablated[at.i2],
      };
      impacts[l][h] = impact;
      opts.onCell?.(impact);
    }
  }

  return {
    impacts,
    baselineLogits,
    baselineTopTokenId: top1,
    baselineTopPiece,
    baselineMargin,
    completed: true,
  };
}

/**
 * Runs a baseline forward pass and an intervened (zero-ablated) forward pass,
 * comparing the resulting output logits and top-5 token predictions.
 * Accepts either a single token or a sequence of tokens to populate KV cache up to pos.
 */
/**
 * Shared metrics for the clean/ablated comparison. Both the sync and async
 * entry points end here so the result panel and circuit map can never
 * disagree about what was measured.
 */
function summarizeComparison(
  baselineLogits: ArrayLike<number>,
  ablatedLogits: ArrayLike<number>,
  targetLayer: number,
  targetHead: number,
  maxPos: number,
  currentToken: number,
  testedPiece: string,
  config: ModelConfig,
  tokenizer: Tokenizer
): AblationComparisonResult {
  const vocabSize = config.vocabSize;

  let maxAbsDiff = 0.0;
  let sumSquaredDiff = 0.0;
  for (let i = 0; i < vocabSize; i++) {
    const diff = ablatedLogits[i] - baselineLogits[i];
    const absDiff = Math.abs(diff);
    if (absDiff > maxAbsDiff) {
      maxAbsDiff = absDiff;
    }
    sumSquaredDiff += diff * diff;
  }
  const l2Diff = Math.sqrt(sumSquaredDiff);

  // Top 5 Baseline
  const baselineRanked: { idx: number; logit: number }[] = [];
  const ablatedRanked: { idx: number; logit: number }[] = [];
  for (let i = 0; i < vocabSize; i++) {
    baselineRanked.push({ idx: i, logit: baselineLogits[i] });
    ablatedRanked.push({ idx: i, logit: ablatedLogits[i] });
  }
  baselineRanked.sort((a, b) => b.logit - a.logit);
  ablatedRanked.sort((a, b) => b.logit - a.logit);

  const baselineTop5: TokenPrediction[] = baselineRanked.slice(0, 5).map((item) => ({
    tokenId: item.idx,
    tokenPiece: tokenizer.decode(currentToken, item.idx),
    logit: item.logit,
  }));

  const ablatedTop5: TokenPrediction[] = ablatedRanked.slice(0, 5).map((item) => ({
    tokenId: item.idx,
    tokenPiece: tokenizer.decode(currentToken, item.idx),
    logit: item.logit,
    delta: item.logit - baselineLogits[item.idx],
  }));

  const baselineTop = baselineTop5[0];
  const ablatedTop = ablatedTop5[0];
  const predictionFlipped = baselineTop.tokenId !== ablatedTop.tokenId;

  // Pre-softmax metrics: the clean run's top-1 token under ablation, and the
  // top1 − top2 margin in each run.
  const baselineTopLogit = baselineLogits[baselineTop.tokenId];
  const ablatedTopTokenLogit = ablatedLogits[baselineTop.tokenId];
  const top1LogitDrop = baselineTopLogit - ablatedTopTokenLogit;
  const baselineMargin = baselineRanked[0].logit - baselineRanked[1].logit;
  const ablatedMargin = ablatedRanked[0].logit - ablatedRanked[1].logit;

  // Softmax probabilities of the baseline top token and top-3 pairs for the
  // plain-English result view (confidence drop + before/after bars).
  const probOf = (logits: ArrayLike<number>, id: number): number => {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
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
    baselineTopLogit,
    ablatedTopTokenLogit,
    top1LogitDrop,
    baselineMargin,
    ablatedMargin,
  };
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

  return summarizeComparison(
    baselineLogits,
    ablatedLogits,
    targetLayer,
    targetHead,
    maxPos,
    currentToken,
    testedPiece,
    config,
    tokenizer
  );
}

/**
 * Non-blocking version of compareHeadAblation: identical metrics, but the two
 * sequence passes yield to the event loop under a time budget. Returns null
 * when cancelled. Pass baselineLogits from a scan to skip the clean run.
 */
export async function compareHeadAblationAsync(
  tokensOrToken: number[] | number,
  pos: number,
  targetLayer: number,
  targetHead: number,
  config: ModelConfig,
  weights: ModelWeights,
  tokenizer: Tokenizer,
  opts: {
    baselineLogits?: ArrayLike<number>;
    isCancelled?: () => boolean;
  } = {}
): Promise<AblationComparisonResult | null> {
  const tokenList = Array.isArray(tokensOrToken) ? tokensOrToken : [tokensOrToken];
  const maxPos = Math.min(pos, tokenList.length - 1);
  const currentToken = tokenList[maxPos];
  const prevToken = maxPos > 0 ? tokenList[maxPos - 1] : 1;
  const testedPiece = tokenizer.decode(prevToken, currentToken);

  const baselineLogits =
    opts.baselineLogits ??
    (await runSequenceLogits(
      tokenList,
      maxPos,
      new HookRegistry(),
      config,
      weights,
      opts.isCancelled
    ));
  if (!baselineLogits || opts.isCancelled?.()) return null;

  const ablatedHooks = new HookRegistry();
  zeroAblateHead(ablatedHooks, targetLayer, targetHead, config.nHeads, config.dim);
  const ablatedLogits = await runSequenceLogits(
    tokenList,
    maxPos,
    ablatedHooks,
    config,
    weights,
    opts.isCancelled
  );
  if (!ablatedLogits) return null;

  return summarizeComparison(
    baselineLogits,
    ablatedLogits,
    targetLayer,
    targetHead,
    maxPos,
    currentToken,
    testedPiece,
    config,
    tokenizer
  );
}
