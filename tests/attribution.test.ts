import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/engine/tensor';
import { attributeStepWrites } from '../src/engine/analysis';
import { rmsnorm } from '../src/engine/ops/rmsnorm';
import { matmul } from '../src/engine/ops/matmul';
import type { StepWrite } from '../src/engine/forward';
import type { ModelWeights } from '../src/engine/model';

const DIM = 8;
const VOCAB = 12;

const wNormArr = Float32Array.from({ length: DIM }, (_, i) => 0.6 + 0.15 * i);
const wClsArr = Float32Array.from({ length: VOCAB * DIM }, (_, i) => Math.sin(i * 7.31) * 1.7);

function mkWeights(): ModelWeights {
  return {
    wcls: new Tensor(wClsArr, [VOCAB, DIM]),
    rmsFinalWeight: new Tensor(wNormArr, [DIM]),
  } as unknown as ModelWeights;
}

function lensLogit(x: Tensor, token: number, w: ModelWeights): number {
  const n = new Tensor(new Float32Array(DIM), [DIM]);
  const out = new Tensor(new Float32Array(VOCAB), [VOCAB]);
  rmsnorm(n, x, w.rmsFinalWeight, DIM);
  matmul(out, n, w.wcls, DIM, VOCAB);
  return out.data[out.offset + token];
}

const vecOf = (fn: (d: number) => number) => Float32Array.from({ length: DIM }, (_, d) => fn(d));

describe('attributeStepWrites — exact logit-space closure', () => {
  const weights = mkWeights();
  const rBefore = vecOf((d) => Math.cos(d * 1.7) * 1.3);

  for (const kind of ['heads', 'ffn'] as const) {
    it(`closes exactly for a ${kind} write (|sum − Δ| < 1e-4)`, () => {
      const vecs =
        kind === 'heads'
          ? [vecOf((d) => 0.3 * d - 0.5), vecOf((d) => -0.2 + 0.1 * d), vecOf(() => 0.4), vecOf((d) => Math.sin(d))]
          : [vecOf((d) => 0.25 * Math.cos(d) - 0.1)];

      const write: StepWrite = { kind, layer: 1, vecs };
      const post = new Tensor(vecOf((d) => rBefore[d] + vecs.reduce((s, v) => s + v[d], 0)), [DIM]);

      const a = 3;
      const b = 7;
      const attrs = attributeStepWrites(write, post, a, b, weights, DIM);
      const attrSum = attrs.reduce((s, x) => s + x.effect, 0);

      const actual =
        lensLogit(post, a, weights) - lensLogit(post, b, weights) -
        (lensLogit(new Tensor(rBefore.slice(), [DIM]), a, weights) -
          lensLogit(new Tensor(rBefore.slice(), [DIM]), b, weights));

      expect(Math.abs(attrSum - actual)).toBeLessThan(1e-4);

      // residual-space closure: Σ vecs must equal the residual delta
      const recovered = vecOf((d) => post.data[post.offset + d] - vecs.reduce((s, v) => s + v[d], 0));
      for (let d = 0; d < DIM; d++) {
        expect(Math.abs(recovered[d] - rBefore[d])).toBeLessThan(1e-6);
      }

      // rescale contributor exists and is a non-head contributor
      const rescale = attrs[attrs.length - 1];
      expect(rescale.kind).toBe('rescale');
      expect(rescale.head).toBe(-1);
    });
  }

  it('attributes ~0 to a synthetic write that raises every vocabulary logit equally', () => {
    // columns are permutations → equal weighted column sums → ones-vec lifts
    // every logit by the same amount, so any A−B comparison sees no change
    const permCls = Float32Array.from({ length: VOCAB * DIM }, (_, i) => {
      const col = Math.floor(i / DIM);
      const d = i % DIM;
      return Math.sin(((col * 3 + d) % DIM) * 5.17 + col) * 0.9;
    });
    const equalWeights = {
      wcls: new Tensor(permCls, [VOCAB, DIM]),
      rmsFinalWeight: new Tensor(Float32Array.from({ length: DIM }, () => 1), [DIM]),
    } as unknown as ModelWeights;

    // check equal column sums under wNorm=1: Σ_d cls[t][d] equal ∀t? ensure by
    // construction: make each column a shifted copy of a base vector
    const base = Float32Array.from({ length: DIM }, (_, d) => Math.sin(d * 2.3) + 0.4);
    const eqCls = Float32Array.from({ length: VOCAB * DIM }, (_, i) => {
      const col = Math.floor(i / DIM);
      const d = i % DIM;
      return base[(d + col) % DIM];
    });
    const w2 = {
      wcls: new Tensor(eqCls, [VOCAB, DIM]),
      rmsFinalWeight: new Tensor(Float32Array.from({ length: DIM }, () => 1), [DIM]),
    } as unknown as ModelWeights;

    const write: StepWrite = { kind: 'heads', layer: 0, vecs: [vecOf(() => 0.5), vecOf(() => 0.5)] };
    const post = new Tensor(vecOf(() => 2), [DIM]); // r_before = ones → dot(g)=0
    const attrs = attributeStepWrites(write, post, 4, 9, w2, DIM);
    const total = attrs.reduce((s, x) => s + x.effect, 0);
    expect(Math.abs(total)).toBeLessThan(1e-5);
  });
});
