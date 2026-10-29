import { describe, it, expect } from 'vitest';
import { Tensor } from '../src/engine/tensor';
import { forwardPass, forward, createRunBuffers } from '../src/engine/forward';
import type { ModelConfig, ModelWeights } from '../src/engine/model';
import { HookRegistry } from '../src/engine/hooks';
import {
  zeroAblateHead,
  meanAblateHead,
  patchActivation,
  addSteeringVector,
} from '../src/engine/interventions';
import { ForwardController } from '../src/bridge/controller';

function createMockModel(): { config: ModelConfig; weights: ModelWeights } {
  const config: ModelConfig = {
    dim: 8,
    hiddenDim: 16,
    nLayers: 2,
    nHeads: 2,
    nKvHeads: 2,
    vocabSize: 4,
    seqLen: 16,
    sharedClassifier: true,
  };

  const weights: ModelWeights = {
    tokenEmbeddingTable: Tensor.zeros([4, 8]),
    rmsAttWeight: [Tensor.fromArray(new Array(8).fill(1)), Tensor.fromArray(new Array(8).fill(1))],
    wq: [Tensor.zeros([8, 8]), Tensor.zeros([8, 8])],
    wk: [Tensor.zeros([8, 8]), Tensor.zeros([8, 8])],
    wv: [Tensor.zeros([8, 8]), Tensor.zeros([8, 8])],
    wo: [Tensor.zeros([8, 8]), Tensor.zeros([8, 8])],
    rmsFfnWeight: [Tensor.fromArray(new Array(8).fill(1)), Tensor.fromArray(new Array(8).fill(1))],
    w1: [Tensor.zeros([16, 8]), Tensor.zeros([16, 8])],
    w2: [Tensor.zeros([8, 16]), Tensor.zeros([8, 16])],
    w3: [Tensor.zeros([16, 8]), Tensor.zeros([16, 8])],
    rmsFinalWeight: Tensor.fromArray(new Array(8).fill(1)),
    wcls: Tensor.zeros([4, 8]),
  };

  // Give token 0 some non-zero embedding values
  for (let i = 0; i < 8; i++) {
    weights.tokenEmbeddingTable.data[i] = i + 1;
    // Set identity diagonal for classifier so output is non-zero
    if (i < 4) {
      weights.wcls.data[i * 8 + i] = 2.0;
    }
  }

  return { config, weights };
}

describe('Stepping (forwardPass generator)', () => {
  it('yields post-attention and post-ffn snapshots for each layer in sequence', () => {
    const { config, weights } = createMockModel();
    const buffers = createRunBuffers(config);
    const gen = forwardPass(0, 0, config, weights, buffers);

    const steps = [];
    let res = gen.next();
    while (!res.done) {
      steps.push(res.value);
      res = gen.next();
    }

    // 2 layers * 2 phases = 4 yields
    expect(steps.length).toBe(4);
    expect(steps[0]).toMatchObject({ layer: 0, phase: 'post-attention' });
    expect(steps[1]).toMatchObject({ layer: 0, phase: 'post-ffn' });
    expect(steps[2]).toMatchObject({ layer: 1, phase: 'post-attention' });
    expect(steps[3]).toMatchObject({ layer: 1, phase: 'post-ffn' });

    // Final result should have logits
    expect(res.value).toHaveProperty('logits');
    expect(res.value.logits.shape).toEqual([config.vocabSize]);
  });
});

describe('ForwardController', () => {
  it('steps and auto-plays to completion with scheduler', async () => {
    const { config, weights } = createMockModel();
    let schedulerCallCount = 0;
    const customScheduler = async () => {
      schedulerCallCount++;
    };

    const controller = new ForwardController(config, weights, new HookRegistry(), customScheduler);
    controller.setStepDelay(0); // tests opt out of autoplay pacing
    controller.start(0, 0);

    expect(controller.state.get().status).toBe('paused');

    // Step once manually
    const hasMore = controller.step();
    expect(hasMore).toBe(true);
    expect(controller.state.get().currentStep?.phase).toBe('post-attention');

    // Auto-play the rest
    const result = await controller.play();
    expect(result).not.toBeNull();
    expect(controller.state.get().status).toBe('completed');
    expect(schedulerCallCount).toBeGreaterThan(0);
  });
});

describe('Interventions', () => {
  it('zeroAblateHead correctly zeroes target head and unregisters cleanly', () => {
    const registry = new HookRegistry();
    const tensor = Tensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8]); // 2 heads of size 4

    const { unregister } = zeroAblateHead(registry, 0, 1, 2, 8); // layer 0, head 1 (index 4..7)

    const res = registry.run('layer.0.attn.head_out', tensor, { layer: 0, position: 0, hookPoint: 'layer.0.attn.head_out' });
    expect(Array.from(res.data)).toEqual([1, 2, 3, 4, 0, 0, 0, 0]);

    unregister();
    const resAfter = registry.run('layer.0.attn.head_out', tensor, { layer: 0, position: 0, hookPoint: 'layer.0.attn.head_out' });
    expect(Array.from(resAfter.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('meanAblateHead clamps head to scalar or tensor mean', () => {
    const registry = new HookRegistry();
    const tensor = Tensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8]);

    // Scalar mean
    meanAblateHead(registry, 0, 0, 0.5, 2, 8);
    const res = registry.run('layer.0.attn.head_out', tensor, { layer: 0, position: 0, hookPoint: 'layer.0.attn.head_out' });
    expect(Array.from(res.data)).toEqual([0.5, 0.5, 0.5, 0.5, 5, 6, 7, 8]);
  });

  it('patchActivation replaces activation with source tensor', () => {
    const registry = new HookRegistry();
    const original = Tensor.fromArray([1, 1, 1, 1]);
    const patch = Tensor.fromArray([9, 9, 9, 9]);

    patchActivation(registry, 0, 'attn.resid', patch);
    const res = registry.run('layer.0.attn.resid', original, { layer: 0, position: 0, hookPoint: 'layer.0.attn.resid' });
    expect(Array.from(res.data)).toEqual([9, 9, 9, 9]);
  });

  it('addSteeringVector adds scaled vector to residual', () => {
    const registry = new HookRegistry();
    const original = Tensor.fromArray([1, 2, 3, 4]);
    const steer = Tensor.fromArray([10, 20, 30, 40]);

    addSteeringVector(registry, 0, 'ffn.resid', steer, 0.5);
    const res = registry.run('layer.0.ffn.resid', original, { layer: 0, position: 0, hookPoint: 'layer.0.ffn.resid' });
    // [1 + 5, 2 + 10, 3 + 15, 4 + 20]
    expect(Array.from(res.data)).toEqual([6, 12, 18, 24]);
  });
});
