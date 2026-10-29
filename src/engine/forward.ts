import { Tensor } from './tensor';
import type { ModelConfig, ModelWeights } from './model';
import { HookRegistry } from './hooks';
import { rmsnorm } from './ops/rmsnorm';
import { matmul } from './ops/matmul';
import { rope } from './ops/rope';
import { softmax } from './ops/softmax';
import { swiglu } from './ops/swiglu';

export interface RunBuffers {
  x: Tensor;      // (dim,)
  xb: Tensor;     // (dim,)
  xb2: Tensor;    // (dim,)
  hb: Tensor;     // (hiddenDim,)
  hb2: Tensor;    // (hiddenDim,)
  q: Tensor;      // (dim,)
  k: Tensor;      // (kvDim,)
  v: Tensor;      // (kvDim,)
  att: Tensor;    // (nHeads, seqLen)
  logits: Tensor; // (vocabSize,)
  // KV Cache: (nLayers, seqLen, kvDim)
  keyCache: Tensor;
  valueCache: Tensor;
}

export function createRunBuffers(config: ModelConfig): RunBuffers {
  const kvDim = Math.floor((config.dim * config.nKvHeads) / config.nHeads);
  return {
    x: Tensor.zeros([config.dim]),
    xb: Tensor.zeros([config.dim]),
    xb2: Tensor.zeros([config.dim]),
    hb: Tensor.zeros([config.hiddenDim]),
    hb2: Tensor.zeros([config.hiddenDim]),
    q: Tensor.zeros([config.dim]),
    k: Tensor.zeros([kvDim]),
    v: Tensor.zeros([kvDim]),
    att: Tensor.zeros([config.nHeads, config.seqLen]),
    logits: Tensor.zeros([config.vocabSize]),
    keyCache: Tensor.zeros([config.nLayers, config.seqLen, kvDim]),
    valueCache: Tensor.zeros([config.nLayers, config.seqLen, kvDim]),
  };
}

export type StepPhase = 'post-attention' | 'post-ffn';

export interface StepState {
  layer: number;
  phase: StepPhase;
  position: number;
  residual: Tensor;             // snapshot clone of current residual x
  attentionScores?: Tensor;     // snapshot clone of attention scores for layer (if post-attention)
}

export interface FinalResult {
  logits: Tensor;
}

/**
 * Synchronous generator function* that steps through the forward pass.
 * Yields a StepState snapshot after each major phase:
 * - 'post-attention' (after layer attention residual addition and hooks)
 * - 'post-ffn' (after layer FFN residual addition and hooks)
 *
 * Returns FinalResult containing the output logits.
 */
export function* forwardPass(
  token: number,
  pos: number,
  config: ModelConfig,
  weights: ModelWeights,
  buffers: RunBuffers,
  hooks: HookRegistry = new HookRegistry()
): Generator<StepState, FinalResult, void> {
  const { dim, hiddenDim, nLayers, nHeads, nKvHeads, seqLen, vocabSize } = config;
  const kvDim = Math.floor((dim * nKvHeads) / nHeads);
  const kvMul = Math.floor(nHeads / nKvHeads);
  const headSize = Math.floor(dim / nHeads);

  // 1. Token embedding lookup: copy row from tokenEmbeddingTable into x
  const embedRow = weights.tokenEmbeddingTable.slice(0, token, token + 1).reshape([dim]);
  buffers.x.data.set(embedRow.data.subarray(embedRow.offset, embedRow.offset + dim), buffers.x.offset);

  // Hook: embed
  let x = hooks.run('embed', buffers.x, { position: pos, hookPoint: 'embed' });
  if (x !== buffers.x) {
    buffers.x.data.set(x.data.subarray(x.offset, x.offset + dim), buffers.x.offset);
    x = buffers.x;
  }

  // Forward all layers
  for (let l = 0; l < nLayers; l++) {
    // 2. Attention RMSNorm: rmsnorm(xb, x, rmsAttWeight[l])
    rmsnorm(buffers.xb, x, weights.rmsAttWeight[l], dim);

    // Hook: layer.N.attn.pre
    const attnPreHook = `layer.${l}.attn.pre`;
    let xb = hooks.run(attnPreHook, buffers.xb, { layer: l, position: pos, hookPoint: attnPreHook });
    if (xb !== buffers.xb) {
      buffers.xb.data.set(xb.data.subarray(xb.offset, xb.offset + dim), buffers.xb.offset);
      xb = buffers.xb;
    }

    // 3. QKV projections:
    matmul(buffers.q, xb, weights.wq[l], dim, dim);
    matmul(buffers.k, xb, weights.wk[l], dim, kvDim);
    matmul(buffers.v, xb, weights.wv[l], dim, kvDim);

    // 4. RoPE on q and k
    rope(buffers.q, buffers.k, pos, dim, kvDim, headSize);

    // Hooks: layer.N.attn.q, layer.N.attn.k, layer.N.attn.v
    const qHook = `layer.${l}.attn.q`;
    const qTensor = hooks.run(qHook, buffers.q, { layer: l, position: pos, hookPoint: qHook });
    if (qTensor !== buffers.q) {
      buffers.q.data.set(qTensor.data.subarray(qTensor.offset, qTensor.offset + dim), buffers.q.offset);
    }

    const kHook = `layer.${l}.attn.k`;
    const kTensor = hooks.run(kHook, buffers.k, { layer: l, position: pos, hookPoint: kHook });
    if (kTensor !== buffers.k) {
      buffers.k.data.set(kTensor.data.subarray(kTensor.offset, kTensor.offset + kvDim), buffers.k.offset);
    }

    const vHook = `layer.${l}.attn.v`;
    const vTensor = hooks.run(vHook, buffers.v, { layer: l, position: pos, hookPoint: vHook });
    if (vTensor !== buffers.v) {
      buffers.v.data.set(vTensor.data.subarray(vTensor.offset, vTensor.offset + kvDim), buffers.v.offset);
    }

    // Save k and v into KV Cache
    const loff = (l * seqLen + pos) * kvDim;
    buffers.keyCache.data.set(
      buffers.k.data.subarray(buffers.k.offset, buffers.k.offset + kvDim),
      buffers.keyCache.offset + loff
    );
    buffers.valueCache.data.set(
      buffers.v.data.subarray(buffers.v.offset, buffers.v.offset + kvDim),
      buffers.valueCache.offset + loff
    );

    // 5. Multihead Attention across all query heads
    const sqrtHeadSize = Math.sqrt(headSize);
    for (let h = 0; h < nHeads; h++) {
      const qHeadOffset = buffers.q.offset + h * headSize;
      const attHeadOffset = buffers.att.offset + h * seqLen;

      // Calculate attention score for each timestep 0..pos
      const kvHeadIdx = Math.floor(h / kvMul);
      for (let t = 0; t <= pos; t++) {
        const kHeadOffset = buffers.keyCache.offset + (l * seqLen + t) * kvDim + kvHeadIdx * headSize;
        let score = 0.0;
        for (let i = 0; i < headSize; i++) {
          score += buffers.q.data[qHeadOffset + i] * buffers.keyCache.data[kHeadOffset + i];
        }
        score /= sqrtHeadSize;
        buffers.att.data[attHeadOffset + t] = score;
      }

      // Softmax over timesteps 0..pos
      const attSlice = new Tensor(buffers.att.data, [pos + 1], [1], attHeadOffset);
      softmax(attSlice, pos + 1);

      // Weighted sum of values into buffers.xb at head slice
      const xbHeadOffset = buffers.xb.offset + h * headSize;
      for (let i = 0; i < headSize; i++) {
        buffers.xb.data[xbHeadOffset + i] = 0.0;
      }

      for (let t = 0; t <= pos; t++) {
        const vHeadOffset = buffers.valueCache.offset + (l * seqLen + t) * kvDim + kvHeadIdx * headSize;
        const a = buffers.att.data[attHeadOffset + t];
        for (let i = 0; i < headSize; i++) {
          buffers.xb.data[xbHeadOffset + i] += a * buffers.valueCache.data[vHeadOffset + i];
        }
      }
    }

    // Hook: layer.N.attn.scores
    const scoresHook = `layer.${l}.attn.scores`;
    const scoresTensor = hooks.run(scoresHook, buffers.att, { layer: l, position: pos, hookPoint: scoresHook });
    if (scoresTensor !== buffers.att) {
      buffers.att.data.set(scoresTensor.data.subarray(scoresTensor.offset, scoresTensor.offset + nHeads * seqLen), buffers.att.offset);
    }

    // Hook: layer.N.attn.head_out
    const headOutHook = `layer.${l}.attn.head_out`;
    xb = hooks.run(headOutHook, buffers.xb, { layer: l, position: pos, hookPoint: headOutHook });
    if (xb !== buffers.xb) {
      buffers.xb.data.set(xb.data.subarray(xb.offset, xb.offset + dim), buffers.xb.offset);
      xb = buffers.xb;
    }

    // 6. Attention output projection Wo @ xb -> xb2
    matmul(buffers.xb2, xb, weights.wo[l], dim, dim);

    // Hook: layer.N.attn.out
    const attnOutHook = `layer.${l}.attn.out`;
    const xb2 = hooks.run(attnOutHook, buffers.xb2, { layer: l, position: pos, hookPoint: attnOutHook });
    if (xb2 !== buffers.xb2) {
      buffers.xb2.data.set(xb2.data.subarray(xb2.offset, xb2.offset + dim), buffers.xb2.offset);
    }

    // 7. Residual connection: x += xb2
    for (let i = 0; i < dim; i++) {
      buffers.x.data[buffers.x.offset + i] += buffers.xb2.data[buffers.xb2.offset + i];
    }

    // Hook: layer.N.attn.resid
    const attnResidHook = `layer.${l}.attn.resid`;
    x = hooks.run(attnResidHook, buffers.x, { layer: l, position: pos, hookPoint: attnResidHook });
    if (x !== buffers.x) {
      buffers.x.data.set(x.data.subarray(x.offset, x.offset + dim), buffers.x.offset);
      x = buffers.x;
    }

    // YIELD post-attention snapshot
    yield {
      layer: l,
      phase: 'post-attention',
      position: pos,
      residual: x.clone(),
      attentionScores: buffers.att.clone(),
    };

    // 8. FFN RMSNorm: rmsnorm(xb, x, rmsFfnWeight[l])
    rmsnorm(buffers.xb, x, weights.rmsFfnWeight[l], dim);

    // Hook: layer.N.ffn.pre
    const ffnPreHook = `layer.${l}.ffn.pre`;
    xb = hooks.run(ffnPreHook, buffers.xb, { layer: l, position: pos, hookPoint: ffnPreHook });
    if (xb !== buffers.xb) {
      buffers.xb.data.set(xb.data.subarray(xb.offset, xb.offset + dim), buffers.xb.offset);
      xb = buffers.xb;
    }

    // 9. FFN projections: w1 @ xb -> hb, w3 @ xb -> hb2
    matmul(buffers.hb, xb, weights.w1[l], dim, hiddenDim);

    // Hook: layer.N.ffn.gate
    const ffnGateHook = `layer.${l}.ffn.gate`;
    let hb = hooks.run(ffnGateHook, buffers.hb, { layer: l, position: pos, hookPoint: ffnGateHook });
    if (hb !== buffers.hb) {
      buffers.hb.data.set(hb.data.subarray(hb.offset, hb.offset + hiddenDim), buffers.hb.offset);
      hb = buffers.hb;
    }

    matmul(buffers.hb2, xb, weights.w3[l], dim, hiddenDim);

    // Hook: layer.N.ffn.up
    const ffnUpHook = `layer.${l}.ffn.up`;
    const hb2 = hooks.run(ffnUpHook, buffers.hb2, { layer: l, position: pos, hookPoint: ffnUpHook });
    if (hb2 !== buffers.hb2) {
      buffers.hb2.data.set(hb2.data.subarray(hb2.offset, hb2.offset + hiddenDim), buffers.hb2.offset);
    }

    // 10. SwiGLU: hb = silu(hb) * hb2
    swiglu(buffers.hb, hb, buffers.hb2, hiddenDim);

    // Hook: layer.N.ffn.swiglu
    const ffnSwigluHook = `layer.${l}.ffn.swiglu`;
    hb = hooks.run(ffnSwigluHook, buffers.hb, { layer: l, position: pos, hookPoint: ffnSwigluHook });
    if (hb !== buffers.hb) {
      buffers.hb.data.set(hb.data.subarray(hb.offset, hb.offset + hiddenDim), buffers.hb.offset);
      hb = buffers.hb;
    }

    // 11. FFN output projection: w2 @ hb -> xb
    matmul(buffers.xb, hb, weights.w2[l], hiddenDim, dim);

    // Hook: layer.N.ffn.out
    const ffnOutHook = `layer.${l}.ffn.out`;
    xb = hooks.run(ffnOutHook, buffers.xb, { layer: l, position: pos, hookPoint: ffnOutHook });
    if (xb !== buffers.xb) {
      buffers.xb.data.set(xb.data.subarray(xb.offset, xb.offset + dim), buffers.xb.offset);
      xb = buffers.xb;
    }

    // 12. Residual connection: x += xb
    for (let i = 0; i < dim; i++) {
      buffers.x.data[buffers.x.offset + i] += buffers.xb.data[buffers.xb.offset + i];
    }

    // Hook: layer.N.ffn.resid
    const ffnResidHook = `layer.${l}.ffn.resid`;
    x = hooks.run(ffnResidHook, buffers.x, { layer: l, position: pos, hookPoint: ffnResidHook });
    if (x !== buffers.x) {
      buffers.x.data.set(x.data.subarray(x.offset, x.offset + dim), buffers.x.offset);
      x = buffers.x;
    }

    // YIELD post-ffn snapshot
    yield {
      layer: l,
      phase: 'post-ffn',
      position: pos,
      residual: x.clone(),
    };
  }

  // 13. Final RMSNorm
  rmsnorm(buffers.x, x, weights.rmsFinalWeight, dim);

  // Hook: final_norm
  x = hooks.run('final_norm', buffers.x, { position: pos, hookPoint: 'final_norm' });
  if (x !== buffers.x) {
    buffers.x.data.set(x.data.subarray(x.offset, x.offset + dim), buffers.x.offset);
    x = buffers.x;
  }

  // 14. Classifier into logits: wcls @ x -> logits
  matmul(buffers.logits, x, weights.wcls, dim, vocabSize);

  // Hook: logits
  const logits = hooks.run('logits', buffers.logits, { position: pos, hookPoint: 'logits' });
  if (logits !== buffers.logits) {
    buffers.logits.data.set(logits.data.subarray(logits.offset, logits.offset + vocabSize), buffers.logits.offset);
  }

  return { logits: buffers.logits };
}

/**
 * Convenience wrapper for running the full forward pass in one shot without stepping.
 */
export function forward(
  token: number,
  pos: number,
  config: ModelConfig,
  weights: ModelWeights,
  buffers: RunBuffers,
  hooks: HookRegistry = new HookRegistry()
): Tensor {
  const gen = forwardPass(token, pos, config, weights, buffers, hooks);
  let res = gen.next();
  while (!res.done) {
    res = gen.next();
  }
  return res.value.logits;
}
