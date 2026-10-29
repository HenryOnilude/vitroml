import type { Tensor } from '../tensor';

/**
 * In-place softmax along the slice of length `size`:
 * Matches llama2.c run.c:
 * void softmax(float* x, int size)
 */
export function softmax(x: Tensor, size: number): void {
  const data = x.data;
  const off = x.offset;

  if (size <= 0) return;

  // Find max for numerical stability
  let maxVal = data[off];
  for (let i = 1; i < size; i++) {
    if (data[off + i] > maxVal) {
      maxVal = data[off + i];
    }
  }

  // Exp and sum
  let sum = 0.0;
  for (let i = 0; i < size; i++) {
    const ev = Math.exp(data[off + i] - maxVal);
    data[off + i] = ev;
    sum += ev;
  }

  // Normalize
  const invSum = 1.0 / sum;
  for (let i = 0; i < size; i++) {
    data[off + i] *= invSum;
  }
}
