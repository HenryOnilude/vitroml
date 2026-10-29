import type { Tensor } from '../tensor';

/**
 * Root Mean Square Layer Normalization (RMSNorm).
 * Matches llama2.c run.c:
 * void rmsnorm(float* o, float* x, float* weight, int size)
 *
 * o = weight * (x / sqrt(mean(x^2) + 1e-5))
 */
export function rmsnorm(o: Tensor, x: Tensor, weight: Tensor, size: number): void {
  const oData = o.data;
  const oOff = o.offset;
  const xData = x.data;
  const xOff = x.offset;
  const wData = weight.data;
  const wOff = weight.offset;

  // Calculate sum of squares
  let ss = 0.0;
  for (let j = 0; j < size; j++) {
    const val = xData[xOff + j];
    ss += val * val;
  }
  ss /= size;
  ss += 1e-5;
  const invRms = 1.0 / Math.sqrt(ss);

  // Normalize and scale
  for (let j = 0; j < size; j++) {
    oData[oOff + j] = wData[wOff + j] * (invRms * xData[xOff + j]);
  }
}
