import { describe, it, expect } from 'vitest';
import { Tensor } from '../src/engine/tensor';
import { matmul } from '../src/engine/ops/matmul';
import { softmax } from '../src/engine/ops/softmax';
import { rmsnorm } from '../src/engine/ops/rmsnorm';
import { rope } from '../src/engine/ops/rope';
import { swiglu } from '../src/engine/ops/swiglu';
import { HookRegistry } from '../src/engine/hooks';

describe('Tensor', () => {
  it('creates contiguous tensor and clones', () => {
    const t = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
    expect(t.shape).toEqual([2, 2]);
    expect(t.strides).toEqual([2, 1]);
    expect(t.size).toBe(4);

    const cloned = t.clone();
    expect(cloned.toJSON()).toEqual({ shape: [2, 2], data: [1, 2, 3, 4] });
    expect(cloned.data).not.toBe(t.data);
  });

  it('slices along dimensions', () => {
    const t = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const row1 = t.slice(0, 1, 2);
    expect(row1.shape).toEqual([1, 3]);
    expect(row1.offset).toBe(3);
    const row1Cloned = row1.clone();
    expect(row1Cloned.toJSON()).toEqual({ shape: [1, 3], data: [4, 5, 6] });
  });

  it('reshapes contiguous tensor', () => {
    const t = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
    const reshaped = t.reshape([4]);
    expect(reshaped.shape).toEqual([4]);
    expect(reshaped.strides).toEqual([1]);
  });
});

describe('Ops', () => {
  it('matmul computes matrix-vector correctly', () => {
    // W = [[1, 2], [3, 4]], x = [5, 6]
    // W @ x = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    const W = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
    const x = Tensor.fromArray([5, 6], [2]);
    const out = Tensor.zeros([2]);
    matmul(out, x, W, 2, 2);
    expect(Array.from(out.data)).toEqual([17, 39]);
  });

  it('softmax normalizes correctly and handles numerical stability', () => {
    const x = Tensor.fromArray([1000, 1000]);
    softmax(x, 2);
    expect(x.data[0]).toBeCloseTo(0.5, 5);
    expect(x.data[1]).toBeCloseTo(0.5, 5);
  });

  it('rmsnorm computes root mean square normalization', () => {
    const x = Tensor.fromArray([2, 2]);
    const w = Tensor.fromArray([1, 1]);
    const o = Tensor.zeros([2]);
    // ss = (4 + 4) / 2 = 4
    // invRms = 1 / sqrt(4 + 1e-5) ~= 0.5
    // o = [2 * 0.5, 2 * 0.5] ~= [1, 1]
    rmsnorm(o, x, w, 2);
    expect(o.data[0]).toBeCloseTo(1.0, 3);
    expect(o.data[1]).toBeCloseTo(1.0, 3);
  });

  it('rope rotates q and k', () => {
    const q = Tensor.fromArray([1, 0, 0, 1]);
    const k = Tensor.fromArray([1, 0, 0, 1]);
    rope(q, k, 0, 4, 4, 2);
    // At pos 0, angle is 0, cos=1, sin=0, so no change
    expect(q.data[0]).toBeCloseTo(1, 5);
    expect(q.data[1]).toBeCloseTo(0, 5);
  });

  it('swiglu computes silu(gate) * up', () => {
    const gate = Tensor.fromArray([0]);
    const up = Tensor.fromArray([5]);
    const out = Tensor.zeros([1]);
    // silu(0) = 0 * 0.5 = 0
    swiglu(out, gate, up, 1);
    expect(out.data[0]).toBe(0);
  });
});

describe('HookRegistry', () => {
  it('registers, runs, mutates and unregisters hooks', () => {
    const registry = new HookRegistry();
    const t = Tensor.fromArray([1, 2]);

    const unreg = registry.register('layer.0.attn.pre', (tensor) => {
      // Mutate
      const mod = tensor.clone();
      mod.data[0] = 99;
      return mod;
    });

    const res = registry.run('layer.0.attn.pre', t, { layer: 0, position: 0, hookPoint: 'layer.0.attn.pre' });
    expect(res.data[0]).toBe(99);

    unreg();
    const res2 = registry.run('layer.0.attn.pre', t, { layer: 0, position: 0, hookPoint: 'layer.0.attn.pre' });
    expect(res2.data[0]).toBe(1);
  });

  it('chains multiple hooks in sequence', () => {
    const registry = new HookRegistry();
    const t = Tensor.fromArray([10]);

    registry.register('test.point', (tensor) => {
      const mod = tensor.clone();
      mod.data[0] += 5;
      return mod;
    });

    registry.register('test.point', (tensor) => {
      const mod = tensor.clone();
      mod.data[0] *= 2;
      return mod;
    });

    const res = registry.run('test.point', t, { position: 0, hookPoint: 'test.point' });
    // (10 + 5) * 2 = 30
    expect(res.data[0]).toBe(30);
  });
});

