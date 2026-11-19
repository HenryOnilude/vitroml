import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { scanHeadImpacts, comparisonCacheKey, type HeadImpact } from '../src/engine/analysis';

// Guards against the seeded cache going stale: src/data/precomputed-scans.json
// is consumed at load time to make the guided head / Act 2 default instant.
// If engine, weights, tokenizer, or generation behaviour change, the stored
// values must be regenerated (npm run precompute) or this test fails.
interface PrecomputedEntry {
  prompt: string;
  tokens: number[];
  pos: number;
  key: string;
  baselineTopTokenId: number;
  baselineTopPiece: string;
  testedPiece: string;
  baselineMargin: number;
  impacts: (HeadImpact | null)[][];
}

const data = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'src/data/precomputed-scans.json'), 'utf8')
) as { entries: PrecomputedEntry[] };

const ckptBuf = fs.readFileSync(path.join(process.cwd(), 'public/models/stories15M.bin'));
const { config, weights } = parseCheckpoint(
  ckptBuf.buffer.slice(ckptBuf.byteOffset, ckptBuf.byteOffset + ckptBuf.byteLength)
);
const tokBuf = fs.readFileSync(path.join(process.cwd(), 'public/models/tokenizer.bin'));
const tokenizer = parseTokenizer(tokBuf.buffer.slice(tokBuf.byteOffset, tokBuf.byteOffset + tokBuf.byteLength), 32000);

// Stored values are exact JSON serialisations of live floats; allow float noise.
const MARGIN_TOL = 1e-3;
const DROP_TOL = 1e-3;

describe('precomputed scans stay in sync with live scanHeadImpacts', () => {
  it('has entries', () => {
    expect(data.entries.length).toBeGreaterThan(0);
  });

  for (const entry of data.entries) {
    it(
      `entry "${entry.prompt}" pos ${entry.pos} matches a live scan`,
      { timeout: 240_000 },
      async () => {
        // The app only hits the seed when this key matches what it computes.
        expect(comparisonCacheKey(entry.tokens, entry.pos)).toBe(entry.key);

        const scan = await scanHeadImpacts(entry.tokens, entry.pos, config, weights, tokenizer);
        expect(scan).not.toBeNull();
        expect(scan!.completed).toBe(true);

        expect(scan!.baselineTopTokenId).toBe(entry.baselineTopTokenId);
        expect(scan!.baselineTopPiece).toBe(entry.baselineTopPiece);
        expect(scan!.testedPiece).toBe(entry.testedPiece);
        expect(Math.abs(scan!.baselineMargin - entry.baselineMargin)).toBeLessThan(MARGIN_TOL);

        for (let l = 0; l < scan!.impacts.length; l++) {
          for (let h = 0; h < scan!.impacts[l].length; h++) {
            const live = scan!.impacts[l][h];
            const stored = entry.impacts[l]?.[h];
            expect(stored, `missing stored impact L${l}:H${h}`).not.toBeNull();
            expect(live, `missing live impact L${l}:H${h}`).not.toBeNull();
            expect(
              Math.abs(live!.top1LogitDrop - stored!.top1LogitDrop),
              `L${l}:H${h} top1LogitDrop drifted: stored ${stored!.top1LogitDrop}, live ${live!.top1LogitDrop}`
            ).toBeLessThan(DROP_TOL);
            expect(live!.predictionFlipped, `L${l}:H${h} flip drifted`).toBe(stored!.predictionFlipped);
            expect(
              Math.abs(live!.marginReduction - stored!.marginReduction),
              `L${l}:H${h} marginReduction drifted: stored ${stored!.marginReduction}, live ${live!.marginReduction}`
            ).toBeLessThan(MARGIN_TOL);
            expect(
              Math.abs(live!.ablatedMargin - stored!.ablatedMargin),
              `L${l}:H${h} ablatedMargin drifted: stored ${stored!.ablatedMargin}, live ${live!.ablatedMargin}`
            ).toBeLessThan(MARGIN_TOL);
          }
        }
      }
    );
  }
});
