/**
 * Ablation sweep: measures zero-ablation impact of every attention head
 * across multiple prompts and positions, and captures each head's attention
 * distribution so claims about head behavior are evidence-based.
 *
 * Usage: npx tsx scripts/ablation_sweep.ts
 * Output: scripts/ablation_sweep_results.json + console summary
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { forward, createRunBuffers } from '../src/engine/forward';
import { HookRegistry } from '../src/engine/hooks';
import { zeroAblateHead } from '../src/engine/interventions';

const PROMPTS = [
  'Once upon a time',
  'The little girl saw a bird',
  'A cat sat on the mat',
  'Ben found a shiny red ball',
  'Mia and her dog went to the park',
  'The sun was very hot today',
  'Tommy lost his toy car',
  'One day a dragon flew over the hill',
  'Anna liked to draw pictures',
  'The boy ate all of his lunch',
];

interface HeadStats {
  l2s: number[];
  l2sNonBos: number[]; // pos > 0 samples only (BOS is the identical computation every prompt)
  top5Hits: number;
  top5HitsNonBos: number;
  attnToBos: number[];  // attention mass on position 0
  attnToPrev: number[]; // attention mass on position-1 (pos>0 samples only)
  attnToSelf: number[]; // attention mass on the current position
  maxAttn: number[];    // largest single attention weight
}

const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const std = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
};

function slice(t: { data: Float32Array; offset: number }, vocab: number) {
  return t.data.subarray(t.offset, t.offset + vocab);
}

function l2Diff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let ss = 0;
  for (let i = 0; i < a.length; i++) {
    const d = b[i] - a[i];
    ss += d * d;
  }
  return Math.sqrt(ss);
}

async function main() {
  const cacheDir = path.join(process.cwd(), '.cache');
  const pick = (name: string) =>
    fs.existsSync(path.join(cacheDir, name))
      ? path.join(cacheDir, name)
      : path.join(process.cwd(), 'public/models', name);
  const read = (p: string) => {
    const b = fs.readFileSync(p);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  };

  const { config, weights } = parseCheckpoint(read(pick('stories15M.bin')));
  const tokenizer = parseTokenizer(read(pick('tokenizer.bin')), config.vocabSize);
  console.log(`model: d=${config.dim} L=${config.nLayers} H=${config.nHeads} vocab=${config.vocabSize}`);

  const headIdx = (l: number, h: number) => l * config.nHeads + h;
  const stats: HeadStats[] = Array.from({ length: config.nLayers * config.nHeads }, () => ({
    l2s: [], l2sNonBos: [], top5Hits: 0, top5HitsNonBos: 0,
    attnToBos: [], attnToPrev: [], attnToSelf: [], maxAttn: [],
  }));

  let samples = 0;
  const t0 = performance.now();

  for (const prompt of PROMPTS) {
    const seq = [1, ...tokenizer.encode(prompt)];
    const positions = [...new Set([0, Math.floor(seq.length / 2), seq.length - 1])];

    for (const pos of positions) {
      const prefix = seq.slice(0, pos + 1);

      // Baseline pass — captures attention rows at the probe position
      const hooks = new HookRegistry();
      const captured = new Float32Array(config.nLayers * config.nHeads * (pos + 1));
      for (let l = 0; l < config.nLayers; l++) {
        hooks.register(`layer.${l}.attn.scores`, (t, ctx) => {
          if (ctx.position !== pos) return;
          for (let h = 0; h < config.nHeads; h++) {
            captured.set(
              t.data.subarray(t.offset + h * config.seqLen, t.offset + h * config.seqLen + pos + 1),
              headIdx(l, h) * (pos + 1)
            );
          }
        });
      }
      const bufs = createRunBuffers(config);
      let logits = forward(prefix[0], 0, config, weights, bufs, hooks);
      for (let p = 1; p <= pos; p++) logits = forward(prefix[p], p, config, weights, bufs, hooks);
      const baseline = Array.from(slice(logits, config.vocabSize));

      // Attention pattern stats
      for (let l = 0; l < config.nLayers; l++) {
        for (let h = 0; h < config.nHeads; h++) {
          const s = stats[headIdx(l, h)];
          const off = headIdx(l, h) * (pos + 1);
          s.attnToBos.push(captured[off]);
          s.attnToSelf.push(captured[off + pos]);
          if (pos > 0) s.attnToPrev.push(captured[off + pos - 1]);
          let m = 0;
          for (let t = 0; t <= pos; t++) if (captured[off + t] > m) m = captured[off + t];
          s.maxAttn.push(m);
        }
      }

      // Ablation runs — one per head
      const l2row: number[] = [];
      for (let l = 0; l < config.nLayers; l++) {
        for (let h = 0; h < config.nHeads; h++) {
          const abuf = createRunBuffers(config);
          const ahooks = new HookRegistry();
          zeroAblateHead(ahooks, l, h, config.nHeads, config.dim);
          let l2logits = forward(prefix[0], 0, config, weights, abuf, ahooks);
          for (let p = 1; p <= pos; p++) l2logits = forward(prefix[p], p, config, weights, abuf, ahooks);
          const d = l2Diff(baseline, slice(l2logits, config.vocabSize));
          stats[headIdx(l, h)].l2s.push(d);
          if (pos > 0) stats[headIdx(l, h)].l2sNonBos.push(d);
          l2row.push(d);
        }
      }

      // top-5 ranking for this sample
      const ranked = l2row.map((d, i) => ({ d, i })).sort((a, b) => b.d - a.d);
      for (const r of ranked.slice(0, 5)) {
        stats[r.i].top5Hits++;
        if (pos > 0) stats[r.i].top5HitsNonBos++;
      }

      samples++;
      const top = ranked[0];
      process.stdout.write(
        `  [${samples}] "${prompt}" pos=${pos} token="${tokenizer.decode(prefix[pos - 1] ?? 1, prefix[pos])}" ` +
          `top head L${Math.floor(top.i / config.nHeads)}:H${top.i % config.nHeads} Δ=${top.d.toFixed(1)}\n`
      );
    }
  }

  console.log(`\n${samples} samples in ${((performance.now() - t0) / 1000).toFixed(0)}s\n`);

  // Summary table
  const summary = stats.map((s, i) => ({
    head: `L${Math.floor(i / config.nHeads)}:H${i % config.nHeads}`,
    layer: Math.floor(i / config.nHeads),
    idx: i,
    meanL2: mean(s.l2s),
    stdL2: std(s.l2s),
    meanL2NonBos: mean(s.l2sNonBos),
    stdL2NonBos: std(s.l2sNonBos),
    top5Pct: (100 * s.top5Hits) / samples,
    top5PctNonBos: (100 * s.top5HitsNonBos) / (samples - PROMPTS.length),
    attnBos: mean(s.attnToBos),
    attnPrev: mean(s.attnToPrev),
    attnSelf: mean(s.attnToSelf),
    maxAttn: mean(s.maxAttn),
  }));
  summary.sort((a, b) => b.meanL2 - a.meanL2);

  console.log('head  meanL2   std    top5%  | non-BOS: meanL2  top5% | attnBOS attnPrev attnSelf maxW');
  for (const s of summary) {
    console.log(
      `${s.head.padEnd(6)}${s.meanL2.toFixed(1).padStart(7)} ${s.stdL2.toFixed(1).padStart(6)} ` +
        `${s.top5Pct.toFixed(0).padStart(5)}% | ${s.meanL2NonBos.toFixed(1).padStart(11)} ${s.top5PctNonBos.toFixed(0).padStart(6)}% | ` +
        `${s.attnBos.toFixed(2).padStart(7)} ${s.attnPrev.toFixed(2).padStart(7)} ` +
        `${s.attnSelf.toFixed(2).padStart(7)} ${s.maxAttn.toFixed(2).padStart(5)}`
    );
  }

  fs.writeFileSync(
    path.join(process.cwd(), 'scripts', 'ablation_sweep_results.json'),
    JSON.stringify({ samples, prompts: PROMPTS, heads: summary }, null, 2)
  );
  console.log('\nwrote scripts/ablation_sweep_results.json');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
