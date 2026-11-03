// Find, for each toolbar preset, the head whose ablation flips the top guess
// at the last prompt position. Fallback: biggest confidence drop on the kept guess.
import { readFileSync } from 'node:fs';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { forward, createRunBuffers } from '../src/engine/forward';
import { HookRegistry } from '../src/engine/hooks';
import { zeroAblateHead } from '../src/engine/interventions';
import { argmaxToken } from '../src/engine/analysis';
import type { Tensor } from '../src/engine/tensor';

const PRESETS = ['Once upon a time', 'The little girl saw a bird'];

const parsed = parseCheckpoint(readFileSync('.cache/stories15M.bin').buffer as ArrayBuffer);
const config = parsed.config;
const weights = parsed.weights;
const tokenizer = parseTokenizer(readFileSync('.cache/tokenizer.bin').buffer as ArrayBuffer, 32000);

function softmaxProb(logits: Tensor, tokenId: number, vocabSize: number): number {
  let max = -Infinity;
  for (let i = 0; i < vocabSize; i++) max = Math.max(max, logits.data[logits.offset + i]);
  let sum = 0;
  for (let i = 0; i < vocabSize; i++) sum += Math.exp(logits.data[logits.offset + i] - max);
  return Math.exp(logits.data[logits.offset + tokenId] - max) / sum;
}

function runSeq(tokens: number[], layer: number | null, head: number | null): Tensor {
  const bufs = createRunBuffers(config);
  const hooks = new HookRegistry();
  if (layer !== null && head !== null) zeroAblateHead(hooks, layer, head, config.nHeads, config.dim);
  let logits!: Tensor;
  for (let p = 0; p < tokens.length; p++) logits = forward(tokens[p], p, config, weights, bufs, hooks);
  return logits;
}

const out: Record<string, { layer: number; head: number; flipped: boolean; note: string }> = {};

for (const prompt of PRESETS) {
  const tokens = tokenizer.encode(prompt);
  const baseLogits = runSeq(tokens, null, null);
  const baseTop = argmaxToken(baseLogits);
  const baseProb = softmaxProb(baseLogits, baseTop, config.vocabSize);
  console.log(`\n"${prompt}" pos=${tokens.length - 1} baseline top=${baseTop} p=${(baseProb * 100).toFixed(1)}%`);

  let bestFlip: { layer: number; head: number; l2: number } | null = null;
  let bestDrop: { layer: number; head: number; drop: number } | null = null;

  for (let l = 0; l < config.nLayers; l++) {
    for (let h = 0; h < config.nHeads; h++) {
      const logits = runSeq(tokens, l, h);
      const top = argmaxToken(logits);
      if (top !== baseTop) {
        let l2 = 0;
        for (let i = 0; i < config.vocabSize; i++) {
          const d = logits.data[logits.offset + i] - baseLogits.data[baseLogits.offset + i];
          l2 += d * d;
        }
        l2 = Math.sqrt(l2);
        if (!bestFlip || l2 > bestFlip.l2) bestFlip = { layer: l, head: h, l2 };
      } else {
        const drop = baseProb - softmaxProb(logits, baseTop, config.vocabSize);
        if (!bestDrop || drop > bestDrop.drop) bestDrop = { layer: l, head: h, drop };
      }
    }
  }

  if (bestFlip) {
    out[prompt] = { layer: bestFlip.layer, head: bestFlip.head, flipped: true, note: `l2=${bestFlip.l2.toFixed(1)}` };
    console.log(`  FLIP: L${bestFlip.layer}:H${bestFlip.head} (l2=${bestFlip.l2.toFixed(1)})`);
  } else if (bestDrop) {
    out[prompt] = { layer: bestDrop.layer, head: bestDrop.head, flipped: false, note: `drop=${(bestDrop.drop * 100).toFixed(1)}%` };
    console.log(`  no flip; biggest confidence drop: L${bestDrop.layer}:H${bestDrop.head} (-${(bestDrop.drop * 100).toFixed(1)}%)`);
  }
}

console.log('\nCURATED =', JSON.stringify(out));
