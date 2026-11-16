<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';

  export let step2Label: string = 'Tap the darkest square in the Circuit Map.';

  const dispatch = createEventDispatcher<{
    guideStep: 1 | 2 | 3;
    tour: void;
  }>();

  // Typewriter preview: the story the model writes, token by token
  const STORY_PIECES = ['there', ' was', ' a', ' little', ' girl', ' named', ' Lily', '.'];
  let writtenCount = 0;
  let writer: ReturnType<typeof setInterval>;
  let hintDismissed = false;
  onMount(() => {
    writer = setInterval(() => {
      writtenCount = (writtenCount + 1) % (STORY_PIECES.length + 3);
    }, 600);
  });
  onDestroy(() => clearInterval(writer));
</script>

<section class="bg-white border border-hairline rounded shadow-subtle px-4 sm:px-5 py-4">
  <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
    <!-- Left: headline + copy + specs -->
    <div class="max-w-xl">
      <div class="text-[9px] font-mono font-bold uppercase tracking-[0.2em] text-accent-bronze mb-1">
        Interactive explorable · client-side
      </div>
      <h1 class="text-2xl sm:text-[2.1rem] sm:leading-[1.15] font-black tracking-tight text-ink font-serif">
        Watch a tiny language model think
      </h1>
      <p class="text-xs text-slate-800 font-medium leading-relaxed mt-1.5 font-sans">
        stories15M is a 15-million-parameter transformer small enough to run entirely in your browser.
        Step through it layer by layer, watch it write a story word by word,
        and switch off attention heads to see how the answer changes.
      </p>
      <div class="mt-2 p-2 bg-amber-50/70 border border-amber-200 rounded text-xs font-sans text-amber-950">
        <strong class="font-semibold text-ink">What is this?</strong> An interactive microscope for a language model — an exploratory instrument, not a data-cleaning or training tool.
      </div>
      <div class="flex flex-wrap items-center gap-2 mt-2.5 text-[10px] font-mono text-ink-muted font-bold">
        <span>15.2M params</span>
        <span class="text-hairline-bright">·</span>
        <span>6 layers × 6 heads</span>
        <span class="text-hairline-bright">·</span>
        <span>288 channels</span>
        <span class="text-hairline-bright">·</span>
        <span>32k words</span>
        <span class="text-hairline-bright">·</span>
        <span>Runs in your browser, nothing is sent to a server</span>
      </div>
    </div>

    <!-- Right: numbered "Try it in 3 steps" interactive strip -->
    <div class="shrink-0 flex flex-col gap-1.5 text-[11px] font-sans font-medium text-ink-secondary min-w-0">
      <div class="text-[9px] font-mono font-bold uppercase tracking-[0.18em] text-ink-muted flex items-center justify-between gap-2">
        <span>Try it in 3 steps</span>
        <button
          type="button"
          on:click={() => dispatch('tour')}
          class="text-[9px] font-mono font-bold uppercase tracking-wider text-accent-bronze hover:text-amber-800 underline underline-offset-2 decoration-amber-300 hover:decoration-amber-600 transition normal-case tracking-normal"
          title="One concept at a time, with simple diagrams"
        >
          or take the 5-slide tour →
        </button>
      </div>
      <button
        type="button"
        on:click={() => dispatch('guideStep', 1)}
        class="flex items-start gap-2 text-left px-2.5 py-1.5 rounded border border-hairline bg-canvas-elevated/70 hover:bg-amber-50/70 hover:border-amber-300 active:scale-[0.99] transition group"
      >
        <span class="font-mono font-bold text-[10px] text-accent-bronze shrink-0 mt-0.5">1.</span>
        <span class="text-slate-900 group-hover:text-ink flex-1">Press Play and watch one word travel through the model.</span>
        <span class="text-ink-muted group-hover:text-accent-bronze text-[10px] font-mono shrink-0">→</span>
      </button>
      <button
        type="button"
        on:click={() => dispatch('guideStep', 2)}
        class="flex items-start gap-2 text-left px-2.5 py-1.5 rounded border border-hairline bg-canvas-elevated/70 hover:bg-amber-50/70 hover:border-amber-300 active:scale-[0.99] transition group"
      >
        <span class="font-mono font-bold text-[10px] text-accent-bronze shrink-0 mt-0.5">2.</span>
        <span class="text-slate-900 group-hover:text-ink flex-1">{step2Label}</span>
        <span class="text-ink-muted group-hover:text-accent-bronze text-[10px] font-mono shrink-0">→</span>
      </button>
      <button
        type="button"
        on:click={() => dispatch('guideStep', 3)}
        class="flex items-start gap-2 text-left px-2.5 py-1.5 rounded border border-hairline bg-canvas-elevated/70 hover:bg-amber-50/70 hover:border-amber-300 active:scale-[0.99] transition group"
      >
        <span class="font-mono font-bold text-[10px] text-accent-bronze shrink-0 mt-0.5">3.</span>
        <span class="text-slate-900 group-hover:text-ink flex-1">Read what changes in the experiment result.</span>
        <span class="text-ink-muted group-hover:text-accent-bronze text-[10px] font-mono shrink-0">→</span>
      </button>
    </div>
  </div>

  <!-- The payoff: the model writing a story, token by token -->
  <div class="relative mt-3.5 pt-3 border-t border-hairline">
    <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] font-mono min-w-0" aria-hidden="true">
      <span class="text-[9px] font-bold uppercase tracking-wider text-ink-muted shrink-0">it writes</span>
      <span class="text-slate-700">⟨s⟩ Once upon a time</span>
      <span class="text-amber-700 font-bold">
        {STORY_PIECES.slice(0, Math.min(writtenCount, STORY_PIECES.length)).join('')}
      </span>
      <span class="hero-caret w-[5px] h-3.5 bg-amber-500"></span>
    </div>

    <!-- First-run nudge: one hint, one action, dismissible -->
    {#if !hintDismissed}
      <div class="mt-2 flex items-center justify-between gap-2 bg-amber-50 border border-amber-200 rounded px-2 py-1">
        <span class="text-[10px] font-sans font-medium text-amber-900">
          First time? Press <strong>▶ Play</strong> in the controls. The model starts writing and everything lights up.
        </span>
        <button
          on:click={() => (hintDismissed = true)}
          class="text-amber-700 hover:text-amber-950 text-[10px] font-mono font-bold px-1"
          aria-label="Dismiss hint"
        >✕</button>
      </div>
    {/if}
  </div>
</section>

<style>
  .hero-caret {
    animation: hero-blink 1.1s steps(2, start) infinite;
  }
  @keyframes hero-blink {
    50% { opacity: 0; }
  }
</style>
