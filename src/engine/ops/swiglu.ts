import type { Tensor } from '../tensor';

/**
 * SwiGLU non-linearity:
 * out = silu(gate) * up
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Writes result into `out` buffer (can be in-place if out === gate).
 * Matches llama2.c run.c:
 * float val = s->hb[i];
 * val *= (1.0f / (1.0f + expf(-val)));
 * val *= s->hb2[i];
 * s->hb[i] = val;
 */
export function swiglu(out: Tensor, gate: Tensor, up: Tensor, size: number): void {
  const outData = out.data;
  const outOff = out.offset;
  const gateData = gate.data;
  const gateOff = gate.offset;
  const upData = up.data;
  const upOff = up.offset;

  for (let i = 0; i < size; i++) {
    const g = gateData[gateOff + i];
    const silu = g / (1.0 + Math.exp(-g));
    outData[outOff + i] = silu * upData[upOff + i];
  }
}
