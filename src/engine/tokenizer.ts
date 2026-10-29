export interface Tokenizer {
  vocabSize: number;
  maxTokenLength: number;
  vocab: string[];
  vocabScores: Float32Array;
  encode(text: string): number[];
  decode(prevToken: number, token: number): string;
}

export function parseTokenizer(buffer: ArrayBuffer, vocabSize: number = 32000): Tokenizer {
  const dataView = new DataView(buffer);
  let offset = 0;

  const maxTokenLength = dataView.getInt32(offset, true);
  offset += 4;

  const vocab: string[] = new Array(vocabSize);
  const vocabScores = new Float32Array(vocabSize);
  const decoder = new TextDecoder('utf-8');

  // Pre-populate byte pieces (for raw byte fallback tokens like in run.c)
  for (let i = 0; i < vocabSize; i++) {
    const score = dataView.getFloat32(offset, true);
    offset += 4;
    vocabScores[i] = score;

    const len = dataView.getInt32(offset, true);
    offset += 4;

    const bytes = new Uint8Array(buffer, offset, len);
    offset += len;

    vocab[i] = decoder.decode(bytes);
  }

  // Build a lookup map from string token to token ID for BPE encoding
  const tokenToId = new Map<string, number>();
  for (let i = 0; i < vocabSize; i++) {
    tokenToId.set(vocab[i], i);
  }

  function decode(prevToken: number, token: number): string {
    if (token < 0 || token >= vocabSize) return '';
    let piece = vocab[token];
    // Following BOS (1), SentencePiece decoder strips leading whitespace
    if (prevToken === 1 && piece.startsWith(' ')) {
      piece = piece.slice(1);
    }
    // Handle raw byte tokens like <0x01>
    const match = piece.match(/^<0x([0-9a-fA-F]{2})>$/);
    if (match) {
      const byteVal = parseInt(match[1], 16);
      piece = String.fromCharCode(byteVal);
    }
    return piece;
  }

  function encode(text: string): number[] {
    // SentencePiece BPE encoding (matching run.c)
    if (text.length === 0) return [];

    // First convert raw text to individual characters / pieces
    const tokens: number[] = [];

    // Add BOS token if standard (BOS = 1 in llama-2)
    // Here we encode the raw text directly:
    // SentencePiece prepends dummy prefix space ' ' (\u2581) to text
    const processedText = ' ' + text.replace(/ /g, ' ');

    for (const char of processedText) {
      const id = tokenToId.get(char);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        // Fallback to byte tokens
        const encoder = new TextEncoder();
        const bytes = encoder.encode(char);
        for (const b of bytes) {
          const hex = b.toString(16).toUpperCase().padStart(2, '0');
          const byteToken = `<0x${hex}>`;
          const byteId = tokenToId.get(byteToken);
          if (byteId !== undefined) {
            tokens.push(byteId);
          }
        }
      }
    }

    // Merge iteratively based on best score
    while (true) {
      let bestScore = -1e10;
      let bestId = -1;
      let bestIdx = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const mergedStr = vocab[tokens[i]] + vocab[tokens[i + 1]];
        const id = tokenToId.get(mergedStr);
        if (id !== undefined) {
          const score = vocabScores[id];
          if (score > bestScore) {
            bestScore = score;
            bestId = id;
            bestIdx = i;
          }
        }
      }

      if (bestIdx === -1) {
        break; // No more merges possible
      }

      // Merge tokens at bestIdx and bestIdx + 1
      tokens[bestIdx] = bestId;
      tokens.splice(bestIdx + 1, 1);
    }

    return tokens;
  }

  return {
    vocabSize,
    maxTokenLength,
    vocab,
    vocabScores,
    encode,
    decode,
  };
}
