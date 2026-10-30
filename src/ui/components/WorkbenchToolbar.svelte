<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Tokenizer } from '../../engine/tokenizer';
  import type { ForwardController } from '../../bridge/controller';

  export let tokenizer: Tokenizer | null = null;
  export let controller: ForwardController | null = null;
  export let promptText: string = 'Once upon a time';
  export let currentPos: number = 0;
  export let generatedTokens: { id: number; piece: string }[] = [];

  const dispatch = createEventDispatcher<{
    tokensChange: { prompt: string; tokens: number[] };
    advance: { resume: 'step' | 'play' };
    userPlay: void;
    userPause: void;
    userStep: void;
    userReset: void;
  }>();

  interface TokenDisplay {
    id: number;
    text: string;
  }

  const SPEEDS = [
    { label: '½×', ms: 2200 },
    { label: '1×', ms: 1100 },
    { label: '2×', ms: 500 },
    { label: '4×', ms: 160 },
  ];
  let speedIdx: number = 1;

  function cycleSpeed() {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    controller?.setStepDelay(SPEEDS[speedIdx].ms);
  }

  let tokenItems: TokenDisplay[] = [];
  let tokenStrip: HTMLElement | null = null;

  // Keep the newest token in view as the stream grows
  $: if (tokenStrip && (generatedTokens.length || tokenItems.length)) {
    tokenStrip.scrollLeft = tokenStrip.scrollWidth;
  }

  $: if (tokenizer && promptText !== undefined) {
    const ids = tokenizer.encode(promptText);
    tokenItems = ids.map((id, index) => {
      const prev = index === 0 ? 1 : ids[index - 1];
      return {
        id,
        text: tokenizer!.decode(prev, id),
      };
    });
    dispatch('tokensChange', { prompt: promptText, tokens: ids });
  }

  let stateStore: ForwardController['state'] | null = null;
  $: stateStore = controller ? controller.state : null;

  function handleStep() {
    if (!controller) return;
    if ($stateStore?.status === 'completed') {
      dispatch('advance', { resume: 'step' });
      return;
    }
    dispatch('userStep');
    if ($stateStore?.status === 'idle') {
      controller.start(1, 0);
    }
    controller.step();
  }

  function handlePlay() {
    if (!controller) return;
    if ($stateStore?.status === 'completed') {
      dispatch('advance', { resume: 'play' });
      return;
    }
    dispatch('userPlay');
    if ($stateStore?.status === 'idle') {
      controller.start(1, 0);
    }
    controller.play();
  }

  function handlePause() {
    if (!controller) return;
    dispatch('userPause');
    controller.pause();
  }

  function handleReset() {
    if (!controller) return;
    dispatch('userReset');
    controller.reset();
    controller.start(1, 0);
  }

  function selectPreset(preset: string) {
    promptText = preset;
  }
</script>

<div class="bg-white border border-hairline rounded shadow-subtle">
  <div class="p-3 flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3">
  <!-- Left: Sleek Prompt Input & Presets -->
  <div class="flex-1 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
    <!-- Sequence Input with High-Contrast Monospace Tag -->
    <div class="relative flex-1 flex items-center bg-slate-50 border border-slate-300 hover:border-slate-400 focus-within:border-ink focus-within:bg-white rounded transition px-2.5 py-1">
      <span class="text-[10px] font-mono font-bold text-white bg-slate-900 px-1.5 py-0.5 rounded uppercase mr-2.5 shrink-0 select-none shadow-sm">
        PROMPT
      </span>
      <input
        id="prompt-input"
        type="text"
        bind:value={promptText}
        placeholder="Type a prompt for the model to continue..."
        class="w-full bg-transparent text-xs text-ink placeholder:text-slate-400 font-sans font-medium outline-none"
      />
    </div>

    <!-- Presets with High-Contrast Typography -->
    <div class="flex items-center gap-1.5 shrink-0">
      <button
        type="button"
        on:click={() => selectPreset('Once upon a time')}
        class="text-xs font-sans text-slate-700 hover:text-ink active:scale-95 px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 border border-slate-300 transition font-semibold"
      >
        "Once upon..."
      </button>
      <button
        type="button"
        on:click={() => selectPreset('The little girl saw a bird')}
        class="text-xs font-sans text-slate-700 hover:text-ink active:scale-95 px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 border border-slate-300 transition font-semibold"
      >
        "The little girl..."
      </button>
    </div>
  </div>

  <!-- Right: Precision Transport Deck -->
  <div class="flex items-center gap-1.5 shrink-0 border-t lg:border-t-0 lg:border-l border-hairline pt-2 lg:pt-0 lg:pl-3">
    <!-- Step Next / Advance Token -->
    <button
      on:click={handleStep}
      disabled={$stateStore?.status === 'running'}
      title="Advance the model one stage so you can watch each part of its work"
      class="px-3.5 py-1.5 bg-white hover:bg-slate-100 active:scale-95 disabled:opacity-40 text-slate-900 border border-slate-300 hover:border-ink rounded text-xs font-bold transition flex items-center gap-1.5 shadow-subtle"
    >
      <svg class="w-3.5 h-3.5 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
      </svg>
      {$stateStore?.status === 'completed' ? 'Next token' : 'Step'}
    </button>

    <!-- Play / Pause -->
    {#if $stateStore?.status === 'running'}
      <button
        on:click={handlePause}
        title="Freeze the run so you can look around"
        class="px-3.5 py-1.5 bg-amber-100 hover:bg-amber-200 active:scale-95 text-amber-950 border border-amber-400 rounded text-xs font-bold transition flex items-center gap-1.5 shadow-subtle"
      >
        <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
        </svg>
        Pause
      </button>
    {:else}
      <button
        on:click={handlePlay}
        title="Run the model forward automatically at the chosen speed"
        class="px-3.5 py-1.5 bg-ink hover:bg-black active:scale-95 disabled:opacity-40 text-white rounded text-xs font-bold transition flex items-center gap-1.5 shadow-sm"
      >
        <svg class="w-3 h-3 fill-current" viewBox="0 0 24 24">
          <path d="M8 5v14l11-7z" />
        </svg>
        {$stateStore?.status === 'completed' ? 'Continue' : 'Play'}
      </button>
    {/if}

    <!-- Play speed -->
    <button
      on:click={cycleSpeed}
      class="w-10 py-1.5 bg-white hover:bg-slate-100 active:scale-95 text-slate-700 hover:text-ink border border-slate-300 hover:border-slate-400 rounded text-xs font-bold transition font-mono shadow-subtle"
      title="Play speed (click to cycle)"
    >
      {SPEEDS[speedIdx].label}
    </button>

    <!-- Reset -->
    <button
      on:click={handleReset}
      class="px-2.5 py-1.5 bg-white hover:bg-slate-100 active:scale-95 text-slate-700 hover:text-ink border border-slate-300 hover:border-slate-400 rounded text-xs font-bold transition shadow-subtle"
      title="Reset back to the start"
    >
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    </button>
  </div>
  </div>

  <div class="px-3 py-1.5 border-t border-hairline text-xs">
    <p class="font-sans font-medium text-slate-900 mb-1">
      Type a story start, then tap <strong>Step</strong> or <strong>Play</strong>.
    </p>
    <details>
      <summary class="cursor-pointer text-[10px] font-sans font-semibold text-ink-muted hover:text-ink select-none">What's this?</summary>
      <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1">
        The controls. Type the start of a story in the box, or tap a preset. <strong>Step</strong> moves the model one stage at a time. <strong>Play</strong> runs it through, and the × button sets the speed. Each chip in the tape is one piece of the story. Slate chips are your words. Amber chips are what the model wrote. The dark chip is the word being worked on right now.
      </p>
    </details>
  </div>

  <!-- Autoregressive Token Stream: BOS → prompt prefill → generated (full-width row, auto-scrolls to newest) -->
  <div class="border-t border-hairline bg-canvas-elevated/60">
    <div
      bind:this={tokenStrip}
      class="flex items-center gap-1 overflow-x-auto px-3 py-1.5 scrollbar-none"
      style="mask-image: linear-gradient(to right, black calc(100% - 56px), transparent); -webkit-mask-image: linear-gradient(to right, black calc(100% - 56px), transparent)"
    >
      <span class="text-[9px] font-mono font-bold text-ink-muted uppercase tracking-wider shrink-0 select-none mr-1" title="Every token the model has seen, in order. Slate = your prompt, amber = what the model wrote.">
        Tokens
      </span>

      <!-- BOS chip at position 0 -->
      <div
        class={`inline-flex items-center px-2 py-0.5 rounded text-xs transition-all shrink-0 ${
          currentPos === 0
            ? 'bg-ink text-white font-bold shadow-sm'
            : 'bg-slate-100 border border-slate-300 text-slate-800 font-medium'
        }`}
        title="Position #0 · start-of-sequence token (ID 1), the model's fixed starting point"
      >
        <span class="font-sans">⟨s⟩</span>
      </div>

      <!-- Prompt tokens (teacher-forced prefill at positions 1..n) -->
      {#each tokenItems as item, idx}
        {@const pos = idx + 1}
        {@const isActive = pos === currentPos}
        <div
          class={`inline-flex items-center px-2 py-0.5 rounded text-xs transition-all shrink-0 ${
            isActive
              ? 'bg-ink text-white font-bold shadow-sm'
              : 'bg-slate-100 border border-slate-300 text-slate-800 hover:border-slate-400 font-medium'
          }`}
          title={`Position #${pos} · your prompt token (ID ${item.id})`}
        >
          <span class="font-sans">{item.text === ' ' ? '␣' : item.text}</span>
        </div>
      {/each}

      <!-- Model-written tokens (sampled argmax output fed back in) -->
      {#each generatedTokens as item, gIdx}
        {@const pos = tokenItems.length + 1 + gIdx}
        {@const isActive = pos === currentPos}
        <div
          class={`inline-flex items-center px-2 py-0.5 rounded text-xs transition-all shrink-0 border animate-token-pop ${
            isActive
              ? 'bg-amber-600 text-white font-bold border-amber-700 shadow-sm'
              : 'bg-amber-50 border-amber-300 text-amber-950 font-semibold'
          }`}
          title={`Position #${pos} · written by the model (ID ${item.id})`}
        >
          <span class="font-sans">{item.piece === ' ' ? '␣' : item.piece}</span>
        </div>
      {/each}
    </div>
  </div>
</div>
