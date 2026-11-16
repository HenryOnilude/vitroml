import type { ForwardController } from '../bridge/controller';

type Dispatch = (event: string, detail?: unknown) => void;

/**
 * Shared transport behaviour for the toolbar and the floating mini deck.
 * Stepping a completed run advances to the next token instead.
 */
export function transportStep(
  controller: ForwardController | null,
  status: string | undefined,
  dispatch: Dispatch,
): void {
  if (!controller) return;
  if (status === 'completed') {
    dispatch('advance', { resume: 'step' });
    return;
  }
  dispatch('userStep');
  if (status === 'idle') controller.start(1, 0);
  controller.step();
}

export function transportPlay(
  controller: ForwardController | null,
  status: string | undefined,
  dispatch: Dispatch,
): void {
  if (!controller) return;
  if (status === 'completed') {
    dispatch('advance', { resume: 'play' });
    return;
  }
  dispatch('userPlay');
  if (status === 'idle') controller.start(1, 0);
  controller.play();
}

export function transportPause(
  controller: ForwardController | null,
  dispatch: Dispatch,
): void {
  if (!controller) return;
  dispatch('userPause');
  controller.pause();
}

export function transportReset(
  controller: ForwardController | null,
  dispatch: Dispatch,
): void {
  if (!controller) return;
  dispatch('userReset');
  controller.reset();
  controller.start(1, 0);
}
