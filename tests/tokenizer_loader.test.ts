import { describe, it, expect } from 'vitest';
import { parseTokenizer } from '../src/engine/tokenizer';
import { parseCheckpoint } from '../src/engine/loader';
import { argmaxToken } from '../src/engine/analysis';
import { Tensor } from '../src/engine/tensor';

// ── Synthetic tokenizer binary (llama2.c format: i32 maxLen, then per token
//    f32 score + i32 len + utf8 bytes) ──────────────────────────────
function buildTokenizerBuffer(vocab: [string, number][]): ArrayBuffer {
  const encoder = new TextEncoder();
  const pieces = vocab.map(([s]) => encoder.encode(s));
  const totalBytes = 4 + vocab.reduce((sum, _, i) => sum + 8 + pieces[i].length, 0);
  const buf = new ArrayBuffer(totalBytes);
  const dv = new DataView(buf);
  let offset = 0;
  dv.setInt32(offset, 8, true);
  offset += 4;
  vocab.forEach(([, score], i) => {
    dv.setFloat32(offset, score, true);
    offset += 4;
    dv.setInt32(offset, pieces[i].length, true);
    offset += 4;
    new Uint8Array(buf, offset, pieces[i].length).set(pieces[i]);
    offset += pieces[i].length;
  });
  return buf;
}

describe('Tokenizer', () => {
  const vocab: [string, number][] = [
    ['<unk>', 0], ['<s>', 0], ['</s>', 0],
    ['a', 0.1], ['b', 0.1], [' ', 0.1],
    ['ab', 0.9], ['<0x41>', 0],
  ];
  const tok = parseTokenizer(buildTokenizerBuffer(vocab), vocab.length);

  it('encodes raw text and merges pairs by score (BPE)', () => {
    // ' a' isn't a merge; 'a'+'b' -> 'ab' (id 6) is
    expect(tok.encode('ab')).toEqual([5, 6]);
  });

  it('falls back to byte tokens for unknown characters', () => {
    // Prefix space (id 5) + 'A' resolves to its byte token <0x41> (id 7)
    expect(tok.encode('A')).toEqual([5, 7]);
  });

  it('strips leading space when decoding after BOS', () => {
    const spTok = parseTokenizer(buildTokenizerBuffer([
      ['<unk>', 0], ['<s>', 0], ['x', 0.1], [' x', 0.5],
    ]), 4);
    expect(spTok.decode(1, 3)).toBe('x');   // ' x' after BOS → 'x'
    expect(spTok.decode(2, 3)).toBe(' x');  // ' x' mid-sequence keeps space
  });

  it('decodes raw byte tokens like <0x41>', () => {
    expect(tok.decode(2, 7)).toBe('A');
  });

  it('returns empty for out-of-range tokens and empty input', () => {
    expect(tok.decode(1, 999)).toBe('');
    expect(tok.encode('')).toEqual([]);
  });
});

// ── Synthetic checkpoint (7×i32 header + f32 weights in llama2.c order) ──
function buildCheckpoint(opts: { vocabSize: number; truncate?: boolean; withFreqs?: boolean }): ArrayBuffer {
  const dim = 8, hiddenDim = 16, nLayers = 2, nHeads = 2, nKvHeads = 2, seqLen = 32;
  const kvDim = (dim * nKvHeads) / nHeads;
  const headSize = dim / nHeads;

  let floats = opts.vocabSize * dim; // token_embedding_table
  floats += nLayers * (dim + dim * dim + kvDim * dim * 2 + dim * dim + dim + hiddenDim * dim * 3);
  floats += dim; // rms_final_weight
  if (opts.withFreqs) floats += seqLen * headSize; // freqs_cos + freqs_imag
  if (opts.vocabSize < 0) floats += Math.abs(opts.vocabSize) * dim; // separate wcls

  const buf = new ArrayBuffer(28 + floats * 4);
  const dv = new DataView(buf);
  const header = [dim, hiddenDim, nLayers, nHeads, nKvHeads, opts.vocabSize, seqLen];
  header.forEach((v, i) => dv.setInt32(i * 4, v, true));

  const out = buf.slice(0, opts.truncate ? buf.byteLength - 4 : buf.byteLength);
  return out;
}

describe('Checkpoint loader', () => {
  it('rejects a buffer smaller than the 28-byte header', () => {
    expect(() => parseCheckpoint(new ArrayBuffer(16))).toThrow(/too small/i);
  });

  it('parses config and marks shared classifier for positive vocab size', () => {
    const { config, weights } = parseCheckpoint(buildCheckpoint({ vocabSize: 16 }));
    expect(config.dim).toBe(8);
    expect(config.nLayers).toBe(2);
    expect(config.vocabSize).toBe(16);
    expect(config.sharedClassifier).toBe(true);
    expect(weights.tokenEmbeddingTable.shape).toEqual([16, 8]);
    expect(weights.wq.length).toBe(2);
  });

  it('skips legacy RoPE freqs tables when present', () => {
    const { config } = parseCheckpoint(buildCheckpoint({ vocabSize: 16, withFreqs: true }));
    expect(config.vocabSize).toBe(16);
  });

  it('rejects a checkpoint truncated inside the weights section', () => {
    expect(() => parseCheckpoint(buildCheckpoint({ vocabSize: 16, truncate: true }))).toThrow(/end of buffer/i);
  });
});

describe('Analysis helpers', () => {
  it('argmaxToken returns the index of the largest logit', () => {
    const t = new Tensor(new Float32Array([0.5, -2, 7.3, 1]), [4]);
    expect(argmaxToken(t)).toBe(2);
  });
});
