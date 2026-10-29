import fs from 'node:fs';
import path from 'node:path';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { compareHeadAblation } from '../src/engine/analysis';

async function main() {
  const cacheDir = path.join(process.cwd(), '.cache');
  const modelPath = path.join(cacheDir, 'stories15M.bin');
  const tokenizerPath = path.join(cacheDir, 'tokenizer.bin');

  const modelBuf = fs.readFileSync(modelPath);
  const modelArrayBuf = modelBuf.buffer.slice(
    modelBuf.byteOffset,
    modelBuf.byteOffset + modelBuf.byteLength
  );
  const { config, weights } = parseCheckpoint(modelArrayBuf);

  const tokBuf = fs.readFileSync(tokenizerPath);
  const tokArrayBuf = tokBuf.buffer.slice(
    tokBuf.byteOffset,
    tokBuf.byteOffset + tokBuf.byteLength
  );
  const tokenizer = parseTokenizer(tokArrayBuf, config.vocabSize);

  const inputToken = 1; // BOS
  const pos = 0;

  console.log('--- Head Ablation Comparison across Layers and Heads ---\n');

  // Baseline vs Layer 0 Head 0 (reference)
  const l0h0 = compareHeadAblation(inputToken, pos, 0, 0, config, weights, tokenizer);
  console.log(`[Reference] Layer 0, Head 0:`);
  console.log(`  L2 norm diff:     ${l0h0.l2Diff.toFixed(4)}`);
  console.log(`  Max abs shift:    ${l0h0.maxAbsDiff.toFixed(4)}`);
  console.log(`  Top prediction:   "${l0h0.ablatedTop5[0].tokenPiece}" (logit ${l0h0.ablatedTop5[0].logit.toFixed(4)}, delta ${l0h0.ablatedTop5[0].delta?.toFixed(4)})`);
  console.log('');

  // Now Layer 5, all heads 0..5
  for (let h = 0; h < config.nHeads; h++) {
    const res = compareHeadAblation(inputToken, pos, 5, h, config, weights, tokenizer);
    console.log(`Layer 5, Head ${h}:`);
    console.log(`  L2 norm diff:     ${res.l2Diff.toFixed(4)}`);
    console.log(`  Max abs shift:    ${res.maxAbsDiff.toFixed(4)}`);
    console.log(`  Top-1 token:      "${res.ablatedTop5[0].tokenPiece}" (logit ${res.ablatedTop5[0].logit.toFixed(4)}, delta ${res.ablatedTop5[0].delta?.toFixed(4)})`);
    console.log(`  Top-2 token:      "${res.ablatedTop5[1].tokenPiece}" (logit ${res.ablatedTop5[1].logit.toFixed(4)}, delta ${res.ablatedTop5[1].delta?.toFixed(4)})`);
    console.log('');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
