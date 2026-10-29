import type { Tensor } from '../tensor';

/**
 * Matrix-vector multiplication: W @ x -> xout
 * W: (d, n) stored in row-major order
 * x: (n,)
 * xout: (d,)
 *
 * Direct match to llama2.c run.c:
 * void matmul(float* xout, float* x, float* w, int n, int d)
 */
export function matmul(xout: Tensor, x: Tensor, w: Tensor, n: number, d: number): void {
  const outData = xout.data;
  const outOff = xout.offset;
  const xData = x.data;
  const xOff = x.offset;
  const wData = w.data;
  const wOff = w.offset;

  for (let i = 0; i < d; i++) {
    let val = 0.0;
    const rowOffset = wOff + i * n;
    for (let j = 0; j < n; j++) {
      val += wData[rowOffset + j] * xData[xOff + j];
    }
    outData[outOff + i] = val;
  }
}
