<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { ForwardController } from '../../bridge/controller';
  import { transportStep, transportPlay, transportPause } from '../transport';

  export let controller: ForwardController | null = null;

  const dispatch = createEventDispatcher<{
    advance: { resume: 'step' | 'play' };
    userPlay: void;
    userPause: void;
    userStep: void;
    userReset: void;
  }>();

  let stateStore: ForwardController['state'] | null = null;
  $: stateStore = controller ? controller.state : null;
  $: status = $stateStore?.status;
  $: stepIndex = $stateStore?.currentStep
    ? $stateStore.currentStep.layer * 2 + ($stateStore.currentStep.phase === 'post-ffn' ? 2 : 1)
    : 0;

  const emit = (t: any, d?: any) => dispatch(t, d);
  const onStep = () => transportStep(controller, status, emit);
  const onPlay = () => transportPlay(controller, status, emit);
  const onPause = () => transportPause(controller, emit);
</script>

<!-- Floating transport: keeps Step/Play reachable after the main controls scroll away -->
<div
  class="fixed bottom-3 inset-x-3 sm:inset-x-auto sm:right-4 z-40 flex items-center justify-center gap-1.5 bg-white/95 backdrop-blur-sm border border-hairline rounded shadow-luxe px-2 py-1.5"
  role="toolbar"
  aria-label="Model controls, compact"
>
  <button
    on:click={onStep}
    disabled={status === 'running'}
    title="Advance the model one stage"
    class="px-3 py-1.5 min-h-[36px] bg-white hover:bg-slate-100 active:bg-slate-200 active:scale-95 disabled:opacity-40 text-slate-900 border border-slate-300 hover:border-ink rounded text-xs font-bold transition flex items-center gap-1.5 shadow-subtle"
  >
    <svg class="w-3 h-3 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
    </svg>
    {status === 'completed' ? 'Next' : 'Step'}
  </button>

  {#if status === 'running'}
    <button
      on:click={onPause}
      title="Freeze the run"
      class="px-3 py-1.5 min-h-[36px] bg-amber-100 hover:bg-amber-200 active:bg-amber-300 active:scale-95 text-amber-950 border border-amber-400 rounded text-xs font-bold transition flex items-center gap-1.5 shadow-subtle"
    >
      <svg class="w-3 h-3 fill-current" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
      Pause
    </button>
  {:else}
    <button
      on:click={onPlay}
      title="Run the model forward automatically"
      class="px-3 py-1.5 min-h-[36px] bg-ink hover:bg-black active:bg-neutral-800 active:scale-95 text-white rounded text-xs font-bold transition flex items-center gap-1.5 shadow-sm"
    >
      <svg class="w-2.5 h-2.5 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
      {status === 'completed' ? 'Continue' : 'Play'}
    </button>
  {/if}

  <span
    class="px-2 py-1.5 text-[10px] font-mono font-bold text-ink-secondary tabular-nums"
    title="Stages of this forward pass completed"
  >
    {stepIndex}<span class="text-ink-faint">/12</span>
  </span>
</div>
