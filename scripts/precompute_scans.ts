// Precompute head-impact scans for the built-in prompts so the Circuit Map,
// the guided Step 2 head, and the Act 2 default are instant on a fresh load.
// Runs the real scanHeadImpacts (no hand-typed values) at each preset's final
// position, plus the shared position-0 scan every load starts on.
// Regenerate after engine/tokenizer changes: npm run precompute
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { mkdirSync } from 'node:fs';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { forward, createRunBuffers } from '../src/engine/forward';
import { HookRegistry } from '../src/engine/hooks';
import {
  argmaxToken,
  scanHeadImpacts,
  comparisonCacheKey,
  type HeadImpactScan,
} from '../src/engine/analysis';
import type { Tensor } from '../src/engine/tensor';

const PRESETS = ['Once upon a time', 'The little girl saw a bird'];
// Mirrors App.svelte generation: argmax continuation until EOS, seqLen, or cap.
const MAX_GENERATED = 16;
const EOS_ID = 2;

const modelPath = existsSync('.cache/stories15M.bin')
  ? '.cache/stories15M.bin'
  : 'public/models/stories15M.bin';
const tokPath = existsSync('.cache/tokenizer.bin')
  ? '.cache/tokenizer.bin'
  : 'public/models/tokenizer.bin';

const parsed = parseCheckpoint(readFileSync(modelPath).buffer as ArrayBuffer);
const config = parsed.config;
const weights = parsed.weights;
const tokenizer = parseTokenizer(readFileSync(tokPath).buffer as ArrayBuffer, 32000);

/** Replays the app's deterministic generation: BOS + prompt, then argmax until EOS/cap. */
function generateSequence(prompt: string): number[] {
  const bufs = createRunBuffers(config);
  const hooks = new HookRegistry();
  const seq = [1, ...tokenizer.encode(prompt)];
  let logits!: Tensor;
  for (let p = 0; p < seq.length; p++) {
    logits = forward(seq[p], p, config, weights, bufs, hooks);
  }
  let generated = 0;
  while (generated < MAX_GENERATED && seq.length < config.seqLen) {
    const next = argmaxToken(logits);
    seq.push(next);
    generated++;
    logits = forward(next, seq.length - 1, config, weights, bufs, hooks);
    if (next === EOS_ID) break;
  }
  return seq;
}

async function scanAt(tokens: number[], pos: number): Promise<HeadImpactScan> {
  const result = await scanHeadImpacts(tokens, pos, config, weights, tokenizer);
  if (!result || !result.completed) throw new Error(`scan did not complete at pos ${pos}`);
  return result;
}

// Only the fields the UI actually reads are stored: the per-head impacts
// (which drive map shading, the "Biggest lever" ring, Step 2 and Act 2's
// preselect) plus scan-level metadata. baselineLogits is deliberately NOT
// written — it costs ~200KB per entry and only saves the result panel one
// ~0.4s baseline pass, which it runs itself.
function toEntry(prompt: string, tokens: number[], pos: number, scan: HeadImpactScan) {
  return {
    prompt,
    tokens,
    pos,
    key: comparisonCacheKey(tokens, pos),
    baselineTopTokenId: scan.baselineTopTokenId,
    baselineTopPiece: scan.baselineTopPiece,
    testedPiece: scan.testedPiece,
    baselineMargin: scan.baselineMargin,
    impacts: scan.impacts,
  };
}

// The app's canonical ranking: a word flip beats everything, then the
// largest margin reduction. Printed for review so regeneration logs show
// which head the UI will highlight.
function topRanked(scan: HeadImpactScan) {
  let best: { l: number; h: number; mred: number; flip: boolean } | null = null;
  for (let l = 0; l < scan.impacts.length; l++) {
    for (let h = 0; h < scan.impacts[l].length; h++) {
      const c = scan.impacts[l][h];
      if (
        c &&
        (!best ||
          (c.predictionFlipped && !best.flip) ||
          (c.predictionFlipped === best.flip && c.marginReduction > best.mred))
      ) {
        best = { l, h, mred: c.marginReduction, flip: c.predictionFlipped };
      }
    }
  }
  return best;
}

const entries: Record<string, unknown>[] = [];

// Position 0 (BOS only): the state every fresh load starts in. One entry
// covers all prompts since the cache key is the token prefix.
{
  const scan = await scanAt([1], 0);
  entries.push(toEntry('(any) position 0', [1], 0, scan));
  const b = topRanked(scan);
  console.log(
    `pos 0 done (key ${comparisonCacheKey([1], 0)}) top L${b!.l}:H${b!.h} mred=${b!.mred.toFixed(2)} flip=${b!.flip}`
  );
}

for (const prompt of PRESETS) {
  const seq = generateSequence(prompt);
  const pos = seq.length - 1;
  console.log(`"${prompt}" -> ${seq.length} tokens, final pos ${pos}`);
  const scan = await scanAt(seq, pos);
  entries.push(toEntry(prompt, seq, pos, scan));
  const b = topRanked(scan);
  console.log(
    `  top head L${b!.l}:H${b!.h} mred=${b!.mred.toFixed(2)} flip=${b!.flip} top="${scan.baselineTopPiece}"`
  );
}

const out = {
  version: 1,
  generatedBy: 'scripts/precompute_scans.ts',
  note: 'Per-(prompt, position) head-impact measurements. Regenerate: npm run precompute',
  entries,
};

mkdirSync('src/data', { recursive: true });
writeFileSync('src/data/precomputed-scans.json', JSON.stringify(out));
console.log(`\nwrote src/data/precomputed-scans.json (${entries.length} entries)`);
