import {
  forwardPass,
  createRunBuffers,
  type StepState,
  type FinalResult,
  type RunBuffers,
} from '../engine/forward';
import type { ModelConfig, ModelWeights } from '../engine/model';
import { HookRegistry } from '../engine/hooks';
import { createObservable, type WritableObservable } from './observable';

export type ExecutionStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

export interface ExecutionState {
  status: ExecutionStatus;
  currentStep: StepState | null;
  result: FinalResult | null;
  error: Error | null;
}

export type Scheduler = () => Promise<void>;

/**
 * Default scheduler trampoline:
 * Uses requestAnimationFrame if in browser window, falls back to setTimeout/setImmediate for Node.js.
 */
export function defaultYieldScheduler(): Promise<void> {
  if (typeof requestAnimationFrame === 'function') {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class ForwardController {
  private config: ModelConfig;
  private weights: ModelWeights;
  private hooks: HookRegistry;
  private buffers: RunBuffers;
  private scheduler: Scheduler;

  private generator: Generator<StepState, FinalResult, void> | null = null;
  private isAutoPlaying: boolean = false;
  private pauseRequested: boolean = false;
  private currentToken: number = 0;
  private currentPos: number = 0;
  // Min pause between yields during autoplay so each phase stays legible.
  private stepDelayMs: number = 1100;

  readonly state: WritableObservable<ExecutionState>;

  constructor(
    config: ModelConfig,
    weights: ModelWeights,
    hooks: HookRegistry = new HookRegistry(),
    scheduler: Scheduler = defaultYieldScheduler
  ) {
    this.config = config;
    this.weights = weights;
    this.hooks = hooks;
    this.scheduler = scheduler;
    this.buffers = createRunBuffers(config);

    this.state = createObservable<ExecutionState>({
      status: 'idle',
      currentStep: null,
      result: null,
      error: null,
    });
  }

  getHookRegistry(): HookRegistry {
    return this.hooks;
  }

  getBuffers(): RunBuffers {
    return this.buffers;
  }

  getToken(): number {
    return this.currentToken;
  }

  getPosition(): number {
    return this.currentPos;
  }

  setStepDelay(ms: number): void {
    this.stepDelayMs = Math.max(0, ms);
  }

  /**
   * Initializes a new forward pass for a given token and position.
   * Starting at position 0 recreates the run buffers (fresh KV cache).
   */
  start(token: number, pos: number = 0): void {
    this.pauseRequested = false;
    this.isAutoPlaying = false;
    this.currentToken = token;
    this.currentPos = pos;
    if (pos === 0) {
      this.buffers = createRunBuffers(this.config);
    }
    this.generator = forwardPass(
      token,
      pos,
      this.config,
      this.weights,
      this.buffers,
      this.hooks
    );
    this.state.set({
      status: 'paused',
      currentStep: null,
      result: null,
      error: null,
    });
  }

  /**
   * Advances the generator by one step (to the next yield point).
   * Returns true if there are more steps remaining, false if finished.
   */
  step(): boolean {
    if (!this.generator) {
      throw new Error('Controller not started. Call start(token, pos) first.');
    }
    const currentStatus = this.state.get().status;
    if (currentStatus === 'completed' || currentStatus === 'error') {
      return false;
    }

    try {
      const res = this.generator.next();
      if (res.done) {
        this.generator = null;
        this.isAutoPlaying = false;
        this.state.update((s) => ({
          ...s,
          status: 'completed',
          result: res.value,
        }));
        return false;
      } else {
        this.state.update((s) => ({
          ...s,
          status: this.isAutoPlaying ? 'running' : 'paused',
          currentStep: res.value,
        }));
        return true;
      }
    } catch (err: unknown) {
      this.generator = null;
      this.isAutoPlaying = false;
      const error = err instanceof Error ? err : new Error(String(err));
      this.state.update((s) => ({
        ...s,
        status: 'error',
        error,
      }));
      throw error;
    }
  }

  /**
   * Continues execution step-by-step using the scheduler trampoline (rAF)
   * until completion or until paused.
   */
  async play(): Promise<FinalResult | null> {
    if (!this.generator) {
      const current = this.state.get();
      return current.result;
    }

    this.isAutoPlaying = true;
    this.pauseRequested = false;
    this.state.update((s) => ({ ...s, status: 'running' }));

    while (this.isAutoPlaying && !this.pauseRequested) {
      const hasMore = this.step();
      if (!hasMore) {
        break;
      }
      // Trampoline to event loop so browser re-renders and handles UI events
      await this.scheduler();
      if (this.stepDelayMs > 0) {
        await new Promise((r) => setTimeout(r, this.stepDelayMs));
      }
    }

    if (this.pauseRequested) {
      this.isAutoPlaying = false;
      this.state.update((s) => ({ ...s, status: 'paused' }));
    }

    return this.state.get().result;
  }

  /**
   * Requests pausing auto-play after the currently evaluating step finishes.
   */
  pause(): void {
    if (this.isAutoPlaying) {
      this.pauseRequested = true;
    }
  }

  /**
   * Autoregressive continuation: feeds a new token at the next position,
   * preserving the KV cache so attention can see all previous positions.
   * Returns false when the context window (seqLen) is exhausted.
   */
  advance(nextToken: number): boolean {
    if (this.currentPos + 1 >= this.config.seqLen) {
      return false;
    }
    this.currentPos += 1;
    this.currentToken = nextToken;
    this.pauseRequested = false;
    this.isAutoPlaying = false;
    this.generator = forwardPass(
      nextToken,
      this.currentPos,
      this.config,
      this.weights,
      this.buffers,
      this.hooks
    );
    this.state.set({
      status: 'paused',
      currentStep: null,
      result: null,
      error: null,
    });
    return true;
  }

  /**
   * Resets execution state and clears current generator.
   */
  reset(): void {
    this.isAutoPlaying = false;
    this.pauseRequested = false;
    this.generator = null;
    this.currentToken = 0;
    this.currentPos = 0;
    this.state.set({
      status: 'idle',
      currentStep: null,
      result: null,
      error: null,
    });
  }
}
