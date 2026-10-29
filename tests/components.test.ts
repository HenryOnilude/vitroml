// @vitest-environment jsdom
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { render } from '@testing-library/svelte';
import { Tensor } from '../src/engine/tensor';
import type { StepState } from '../src/engine/forward';
import type { ExecutionState } from '../src/bridge/controller';
import AttentionMap from '../src/ui/components/AttentionMap.svelte';
import NarrativeBar from '../src/ui/components/NarrativeBar.svelte';

// Svelte transitions (fly) require requestAnimationFrame; jsdom may not provide it
beforeAll(() => {
  if (typeof globalThis.requestAnimationFrame !== 'function') {
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) =>
      setTimeout(() => cb(performance.now()), 0)
    );
  }
});

function makeStepState(overrides: Partial<StepState> = {}): StepState {
  // 2 heads × seqLen 4 attention field, position 1 (2 visible columns)
  const attentionScores = new Tensor(
    new Float32Array([
      0.8, 0.2, 0, 0, // head 0 focuses pos 0
      0.1, 0.9, 0, 0, // head 1 focuses pos 1
    ]),
    [2, 4]
  );
  return {
    layer: 0,
    phase: 'post-attention',
    position: 1,
    residual: Tensor.zeros([8]),
    attentionScores,
    ...overrides,
  };
}

function makeExecState(overrides: Partial<ExecutionState> = {}): ExecutionState {
  return {
    status: 'running',
    currentStep: makeStepState(),
    result: null,
    error: null,
    ...overrides,
  };
}

describe('AttentionMap component', () => {
  it('shows a self-explanatory empty state before any attention pass', () => {
    const { getByText } = render(AttentionMap, { props: { stepState: null } });
    getByText(/Nothing here yet/);
    getByText(/waiting for the first attention pass/);
  });

  it('renders one row per head with correct focus annotations', () => {
    const { getByText, getAllByText, container } = render(AttentionMap, {
      props: {
        stepState: makeStepState(),
        positionLabels: [
          { text: '<s>', id: 1 },
          { text: ' Once', id: 42 },
        ],
      },
    });
    getByText('H0');
    getByText('H1');
    getByText(/looks back over 2 tokens/);
    // Head 0's favorite is position 0 ("<s>" trims to empty → falls back to #1);
    // head 1's favorite is position 1 → "Once"
    const rows = container.querySelectorAll('.space-y-1 > div');
    expect(rows.length).toBe(3); // 2 head rows + label row
    // "Once" is head 1's top focus AND the position-1 axis label
    expect(getAllByText('"Once"').length).toBeGreaterThanOrEqual(2);
  });

  it('dims non-highlighted rows when a head is highlighted', () => {
    const { container } = render(AttentionMap, {
      props: { stepState: makeStepState(), highlightedHead: 1 },
    });
    const headRows = container.querySelectorAll('.space-y-1 > div');
    expect(headRows[0].className).toContain('opacity-30');
    expect(headRows[1].className).not.toContain('opacity-30');
  });
});

describe('NarrativeBar component', () => {
  it('explains the idle state in plain language', () => {
    const { getByText } = render(NarrativeBar, { props: { state: null } });
    getByText('Token Ready at the Input');
    getByText(/Press "Step" or "Play"/);
  });

  it('narrates the layer-0 attention phase', () => {
    const { getByText } = render(NarrativeBar, {
      props: { state: makeExecState() },
    });
    getByText('Heads Gather Nearby Context');
    getByText(/LAYER 0 · ATTENTION/);
  });

  it('names the sampled token on completion', () => {
    const { getByText } = render(NarrativeBar, {
      props: {
        state: makeExecState({ status: 'completed', currentStep: null }),
        sampledPiece: ' upon',
        currentPos: 1,
      },
    });
    getByText(/Model Chose " upon"/);
  });

  it('treats a running state without a step as queued', () => {
    const { getByText } = render(NarrativeBar, {
      props: { state: makeExecState({ currentStep: null }), currentToken: 7, currentPos: 3 },
    });
    getByText(/Token 7 Queued at Position 3/);
  });
});
