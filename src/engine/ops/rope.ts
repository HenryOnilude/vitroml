import type { Tensor } from '../tensor';

/**
 * Rotary Positional Encoding (RoPE) applied to query and key vectors.
 * Matches llama2.c run.c:
 * Rotates pairs of elements in each head according to their frequency and position.
 *
 * @param q Query tensor (dim,)
 * @param k Key tensor (kvDim,)
 * @param pos Current sequence position
 * @param dim Transformer dimension (e.g. 288)
 * @param kvDim Key/value dimension (dim * n_kv_heads / n_heads)
 * @param headSize Dimension per head (dim / n_heads)
 */
export function rope(
  q: Tensor,
  k: Tensor,
  pos: number,
  dim: number,
  kvDim: number,
  headSize: number
): void {
  const qData = q.data;
  const qOff = q.offset;
  const kData = k.data;
  const kOff = k.offset;

  for (let i = 0; i < dim; i += 2) {
    const headDim = i % headSize;
    const freq = 1.0 / Math.pow(10000.0, headDim / headSize);
    const val = pos * freq;
    const fcr = Math.cos(val);
    const fci = Math.sin(val);

    // Rotate q
    const q0 = qData[qOff + i];
    const q1 = qData[qOff + i + 1];
    qData[qOff + i] = q0 * fcr - q1 * fci;
    qData[qOff + i + 1] = q0 * fci + q1 * fcr;

    // Rotate k if i < kvDim
    if (i < kvDim) {
      const k0 = kData[kOff + i];
      const k1 = kData[kOff + i + 1];
      kData[kOff + i] = k0 * fcr - k1 * fci;
      kData[kOff + i + 1] = k0 * fci + k1 * fcr;
    }
  }
}
