import type { Tensor } from './tensor';

export interface HookContext {
  layer?: number;
  position: number;
  hookPoint: string;
}

export type HookFn = (tensor: Tensor, ctx: HookContext) => Tensor | void;

export class HookRegistry {
  private hooks = new Map<string, HookFn[]>();

  register(point: string, fn: HookFn): () => void {
    let list = this.hooks.get(point);
    if (!list) {
      list = [];
      this.hooks.set(point, list);
    }
    list.push(fn);

    return () => {
      const currentList = this.hooks.get(point);
      if (!currentList) return;
      const idx = currentList.indexOf(fn);
      if (idx !== -1) {
        currentList.splice(idx, 1);
      }
      if (currentList.length === 0) {
        this.hooks.delete(point);
      }
    };
  }

  run(point: string, tensor: Tensor, ctx: HookContext): Tensor {
    const list = this.hooks.get(point);
    if (!list || list.length === 0) {
      return tensor;
    }

    let current = tensor;
    for (const fn of list) {
      const result = fn(current, ctx);
      if (result !== undefined) {
        current = result;
      }
    }
    return current;
  }

  clear(): void {
    this.hooks.clear();
  }

  has(point: string): boolean {
    const list = this.hooks.get(point);
    return !!list && list.length > 0;
  }
}
