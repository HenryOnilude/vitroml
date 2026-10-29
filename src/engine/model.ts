import type { Tensor } from './tensor';

export interface ModelConfig {
  dim: number;         // transformer dimension (288)
  hiddenDim: number;   // ffn hidden dimension (768)
  nLayers: number;     // number of layers (6)
  nHeads: number;      // number of query heads (6)
  nKvHeads: number;    // number of kv heads (6)
  vocabSize: number;   // vocabulary size (32000)
  seqLen: number;      // max sequence length (256)
  sharedClassifier: boolean;
}

export interface ModelWeights {
  tokenEmbeddingTable: Tensor; // (vocabSize, dim)
  rmsAttWeight: Tensor[];       // nLayers tensors of shape (dim,)
  wq: Tensor[];                 // nLayers tensors of shape (dim, dim)
  wk: Tensor[];                 // nLayers tensors of shape (kvDim, dim)
  wv: Tensor[];                 // nLayers tensors of shape (kvDim, dim)
  wo: Tensor[];                 // nLayers tensors of shape (dim, dim)
  rmsFfnWeight: Tensor[];       // nLayers tensors of shape (dim,)
  w1: Tensor[];                 // nLayers tensors of shape (hiddenDim, dim)
  w2: Tensor[];                 // nLayers tensors of shape (dim, hiddenDim)
  w3: Tensor[];                 // nLayers tensors of shape (hiddenDim, dim)
  rmsFinalWeight: Tensor;      // (dim,)
  wcls: Tensor;                 // (vocabSize, dim)
}
