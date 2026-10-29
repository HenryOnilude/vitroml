import { Tensor } from './tensor';
import type { HookFn, HookRegistry } from './hooks';

export interface InterventionRegistration {
  point: string;
  hook: HookFn;
  unregister: () => void;
}

/**
 * Creates a HookFn that zeros out a specific attention head's output at `layer.L.attn.head_out`.
 * Head tensor slice in head_out has length `headDim = dim / nHeads`, starting at `head * headDim`.
 */
export function createZeroAblateHeadHook(
  head: number,
  nHeads: number = 6,
  dim: number = 288
): HookFn {
  const headDim = Math.floor(dim / nHeads);
  const headStart = head * headDim;
  const headEnd = headStart + headDim;

  return (tensor: Tensor) => {
    // Clone to avoid mutating original shared buffer unexpectedly if caller expects pure hook
    const modified = tensor.clone();
    for (let i = headStart; i < headEnd; i++) {
      modified.data[modified.offset + i] = 0.0;
    }
    return modified;
  };
}

/**
 * Helper to register zeroAblateHead directly into a HookRegistry.
 */
export function zeroAblateHead(
  registry: HookRegistry,
  layer: number,
  head: number,
  nHeads: number = 6,
  dim: number = 288
): InterventionRegistration {
  const point = `layer.${layer}.attn.head_out`;
  const hook = createZeroAblateHeadHook(head, nHeads, dim);
  const unregister = registry.register(point, hook);
  return { point, hook, unregister };
}

/**
 * Creates a HookFn that replaces a specific attention head's output with a precomputed mean vector or scalar.
 */
export function createMeanAblateHeadHook(
  head: number,
  mean: Tensor | number,
  nHeads: number = 6,
  dim: number = 288
): HookFn {
  const headDim = Math.floor(dim / nHeads);
  const headStart = head * headDim;
  const headEnd = headStart + headDim;

  return (tensor: Tensor) => {
    const modified = tensor.clone();
    if (typeof mean === 'number') {
      for (let i = headStart; i < headEnd; i++) {
        modified.data[modified.offset + i] = mean;
      }
    } else {
      for (let i = 0; i < headDim; i++) {
        modified.data[modified.offset + headStart + i] = mean.data[mean.offset + i];
      }
    }
    return modified;
  };
}

/**
 * Helper to register meanAblateHead directly into a HookRegistry.
 */
export function meanAblateHead(
  registry: HookRegistry,
  layer: number,
  head: number,
  mean: Tensor | number,
  nHeads: number = 6,
  dim: number = 288
): InterventionRegistration {
  const point = `layer.${layer}.attn.head_out`;
  const hook = createMeanAblateHeadHook(head, mean, nHeads, dim);
  const unregister = registry.register(point, hook);
  return { point, hook, unregister };
}

/**
 * Creates a HookFn that replaces an activation tensor with one captured from a different run.
 * Accepts full hook point like "layer.2.attn.resid" or prefix "attn.resid" with layer passed to helper.
 */
export function createPatchActivationHook(sourceTensor: Tensor): HookFn {
  return () => {
    return sourceTensor.clone();
  };
}

/**
 * Helper to register patchActivation directly into a HookRegistry.
 * If point starts with "layer.", it is used directly; otherwise it is formatted as `layer.${layer}.${point}`.
 */
export function patchActivation(
  registry: HookRegistry,
  layer: number,
  point: string,
  sourceTensor: Tensor
): InterventionRegistration {
  const fullPoint = point.startsWith('layer.') ? point : `layer.${layer}.${point}`;
  const hook = createPatchActivationHook(sourceTensor);
  const unregister = registry.register(fullPoint, hook);
  return { point: fullPoint, hook, unregister };
}

/**
 * Creates a HookFn that adds alpha * vector to the activation/residual tensor.
 */
export function createAddSteeringVectorHook(vector: Tensor, alpha: number = 1.0): HookFn {
  return (tensor: Tensor) => {
    const modified = tensor.clone();
    const size = Math.min(modified.size, vector.size);
    for (let i = 0; i < size; i++) {
      modified.data[modified.offset + i] += alpha * vector.data[vector.offset + i];
    }
    return modified;
  };
}

/**
 * Helper to register addSteeringVector directly into a HookRegistry.
 * If point starts with "layer." or "embed" or "final_norm", it is used directly; otherwise formatted as `layer.${layer}.${point}`.
 */
export function addSteeringVector(
  registry: HookRegistry,
  layer: number,
  point: string,
  vector: Tensor,
  alpha: number = 1.0
): InterventionRegistration {
  const fullPoint =
    point.startsWith('layer.') || point === 'embed' || point === 'final_norm'
      ? point
      : `layer.${layer}.${point}`;
  const hook = createAddSteeringVectorHook(vector, alpha);
  const unregister = registry.register(fullPoint, hook);
  return { point: fullPoint, hook, unregister };
}
