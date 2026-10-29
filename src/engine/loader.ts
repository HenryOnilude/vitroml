import { Tensor } from './tensor';
import type { ModelConfig, ModelWeights } from './model';

export function assertLittleEndian(): void {
  const testBuf = new ArrayBuffer(2);
  const u8 = new Uint8Array(testBuf);
  const u16 = new Uint16Array(testBuf);
  u16[0] = 0x0102;
  // In little-endian, least significant byte (0x02) comes first
  if (u8[0] !== 0x02) {
    throw new Error('Big-endian platform detected; checkpoint loader requires little-endian.');
  }
}

export function parseCheckpoint(buffer: ArrayBuffer): { config: ModelConfig; weights: ModelWeights } {
  assertLittleEndian();

  if (buffer.byteLength < 28) {
    throw new Error(`Buffer too small for llama2.c header: ${buffer.byteLength} bytes`);
  }

  const header = new DataView(buffer, 0, 28);
  const dim = header.getInt32(0, true);
  const hiddenDim = header.getInt32(4, true);
  const nLayers = header.getInt32(8, true);
  const nHeads = header.getInt32(12, true);
  const nKvHeads = header.getInt32(16, true);
  const rawVocabSize = header.getInt32(20, true);
  const seqLen = header.getInt32(24, true);

  const sharedClassifier = rawVocabSize > 0;
  const vocabSize = Math.abs(rawVocabSize);

  const config: ModelConfig = {
    dim,
    hiddenDim,
    nLayers,
    nHeads,
    nKvHeads,
    vocabSize,
    seqLen,
    sharedClassifier,
  };

  const headSize = Math.floor(dim / nHeads);
  const kvDim = Math.floor((dim * nKvHeads) / nHeads);

  // Weights start immediately after the 28-byte header
  // 28 is 4-byte aligned (28 / 4 = 7 floats)
  let floatOffset = 28 / 4;

  function takeFloat32View(shape: number[]): Tensor {
    let size = 1;
    for (const d of shape) size *= d;
    const byteOffset = floatOffset * 4;
    if (byteOffset + size * 4 > buffer.byteLength) {
      throw new Error(
        `Unexpected end of buffer: needed ${size * 4} bytes at offset ${byteOffset}, buffer length is ${buffer.byteLength}`
      );
    }
    const data = new Float32Array(buffer, byteOffset, size);
    floatOffset += size;
    return new Tensor(data, shape);
  }

  // 1. token_embedding_table: (vocabSize, dim)
  const tokenEmbeddingTable = takeFloat32View([vocabSize, dim]);

  // 2. rms_att_weight: nLayers x (dim,)
  const rmsAttWeight: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    rmsAttWeight.push(takeFloat32View([dim]));
  }

  // 3. wq: nLayers x (dim, dim)
  const wq: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    wq.push(takeFloat32View([dim, dim]));
  }

  // 4. wk: nLayers x (kvDim, dim) - note: run.c layout has kvDim rows of dim
  const wk: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    wk.push(takeFloat32View([kvDim, dim]));
  }

  // 5. wv: nLayers x (kvDim, dim)
  const wv: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    wv.push(takeFloat32View([kvDim, dim]));
  }

  // 6. wo: nLayers x (dim, dim)
  const wo: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    wo.push(takeFloat32View([dim, dim]));
  }

  // 7. rms_ffn_weight: nLayers x (dim,)
  const rmsFfnWeight: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    rmsFfnWeight.push(takeFloat32View([dim]));
  }

  // 8. w1: nLayers x (hiddenDim, dim)
  const w1: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    w1.push(takeFloat32View([hiddenDim, dim]));
  }

  // 9. w2: nLayers x (dim, hiddenDim)
  const w2: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    w2.push(takeFloat32View([dim, hiddenDim]));
  }

  // 10. w3: nLayers x (hiddenDim, dim)
  const w3: Tensor[] = [];
  for (let l = 0; l < nLayers; l++) {
    w3.push(takeFloat32View([hiddenDim, dim]));
  }

  // 11. rms_final_weight: (dim,)
  const rmsFinalWeight = takeFloat32View([dim]);

  // 12. Skip freqs_cos and freqs_sin if present (legacy llama2.c export included them)
  // In run.c:
  // ptr += p->seq_len * head_size / 2; // freq_cis_real
  // ptr += p->seq_len * head_size / 2; // freq_cis_imag
  const freqsSize = (seqLen * headSize) / 2;
  const remainingFloats = (buffer.byteLength - floatOffset * 4) / 4;

  // If there are extra floats corresponding to RoPE freqs table:
  if (remainingFloats >= freqsSize * 2) {
    floatOffset += freqsSize * 2;
  }

  // 13. Classifier weights (wcls)
  let wcls: Tensor;
  if (sharedClassifier) {
    wcls = tokenEmbeddingTable;
  } else {
    wcls = takeFloat32View([vocabSize, dim]);
  }

  return {
    config,
    weights: {
      tokenEmbeddingTable,
      rmsAttWeight,
      wq,
      wk,
      wv,
      wo,
      rmsFfnWeight,
      w1,
      w2,
      w3,
      rmsFinalWeight,
      wcls,
    },
  };
}
