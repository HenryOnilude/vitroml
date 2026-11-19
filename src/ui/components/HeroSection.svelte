<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';

  export let step2Label: string = 'Tap the darkest square in the Circuit Map.';
  // False until the model has generated at least one word — keeps a first-time
  // visitor from running their first experiment at position 0 (BOS only),
  // where every head's effect is near zero.
  export let step2Enabled: boolean = true;

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
      <div class="mt-2 p-2 bg-amber-50/70 border border-amber-200 rounded text-xs font-sans text-amber-950 leading-relaxed">
        <strong class="font-semibold text-ink">What is this?</strong> A tiny AI storyteller, technically a 15M-parameter transformer language model, running entirely in your browser. Press Play and watch it write one word at a time (a forward pass), look inside its 6 layers to see how the 36 attention heads pick each word, then switch one head off (ablation) and watch the answer change. It is for exploring how AI thinks.
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
      <p class="text-[10px] text-ink-muted leading-snug -mt-0.5">
        The whole idea in three clicks: run it, break one part, see what changed.
      </p>
      <button
        type="button"
        on:click={() => dispatch('guideStep', 1)}
        class="flex items-start gap-2 text-left px-2.5 py-1.5 rounded border border-hairline bg-canvas-elevated/70 hover:bg-amber-50/70 hover:border-amber-300 active:scale-[0.99] transition group"
      >
        <span class="font-mono font-bold text-[10px] text-accent-bronze shrink-0 mt-0.5">1.</span>
        <span class="flex-1 min-w-0">
          <span class="block text-slate-900 group-hover:text-ink font-semibold">Press Play in the controls below.</span>
          <span class="block text-[10px] text-ink-muted leading-snug mt-0.5">The model guesses the next word. Watch it travel through all 6 layers as each part lights up.</span>
        </span>
        <span class="text-ink-muted group-hover:text-accent-bronze text-[10px] font-mono shrink-0 mt-0.5">→</span>
      </button>
      <button
        type="button"
        on:click={() => dispatch('guideStep', 2)}
        disabled={!step2Enabled}
        title={step2Enabled ? undefined : 'Finish the story first — press Play so the model has written at least one word'}
        class={`flex items-start gap-2 text-left px-2.5 py-1.5 rounded border border-hairline transition group ${
          step2Enabled
            ? 'bg-canvas-elevated/70 hover:bg-amber-50/70 hover:border-amber-300 active:scale-[0.99] cursor-pointer'
            : 'bg-canvas-elevated/40 opacity-60 cursor-not-allowed'
        }`}
      >
        <span class="font-mono font-bold text-[10px] text-accent-bronze shrink-0 mt-0.5">2.</span>
        <span class="flex-1 min-w-0">
          <span class="block text-slate-900 group-hover:text-ink font-semibold">
            {step2Enabled ? step2Label : 'Finish the story first — press Play.'}
          </span>
          <span class="block text-[10px] text-ink-muted leading-snug mt-0.5">Attention heads are small parts that look back at earlier words. L is the layer number, H is the head inside it.</span>
        </span>
        <span class="text-ink-muted group-hover:text-accent-bronze text-[10px] font-mono shrink-0 mt-0.5">→</span>
      </button>
      <button
        type="button"
        on:click={() => dispatch('guideStep', 3)}
        class="flex items-start gap-2 text-left px-2.5 py-1.5 rounded border border-hairline bg-canvas-elevated/70 hover:bg-amber-50/70 hover:border-amber-300 active:scale-[0.99] transition group"
      >
        <span class="font-mono font-bold text-[10px] text-accent-bronze shrink-0 mt-0.5">3.</span>
        <span class="flex-1 min-w-0">
          <span class="block text-slate-900 group-hover:text-ink font-semibold">Read what changed in the experiment result.</span>
          <span class="block text-[10px] text-ink-muted leading-snug mt-0.5">The guess before and after, side by side. If the word or its confidence moved, that head was doing real work.</span>
        </span>
        <span class="text-ink-muted group-hover:text-accent-bronze text-[10px] font-mono shrink-0 mt-0.5">→</span>
      </button>
    </div>
  </div>

  <!-- Cover diagram: the whole story in one annotated picture -->
  <div class="mt-3.5 pt-3 border-t border-hairline overflow-x-auto scrollbar-none">
    <svg
      viewBox="0 0 1000 168"
      class="w-full min-w-[720px] h-auto"
      role="img"
      aria-label="Diagram: your words enter the model, attention heads look back inside 6 layers, a next-word guess comes out, and switching a head off changes that guess"
    >
      <!-- scene captions -->
      <text x="20" y="14" font-size="9" font-family="monospace" font-weight="700" fill="#b45309">1 · GIVE IT WORDS</text>
      <text x="370" y="14" font-size="9" font-family="monospace" font-weight="700" fill="#b45309">2 · WATCH IT THINK</text>
      <text x="768" y="14" font-size="9" font-family="monospace" font-weight="700" fill="#b45309">3 · BREAK IT</text>

      <!-- scene 1: input tokens -->
      <rect x="20" y="42" width="44" height="22" rx="4" fill="#1e293b" /><text x="42" y="57" text-anchor="middle" font-size="10" font-family="sans-serif" fill="white">Once</text>
      <rect x="70" y="42" width="44" height="22" rx="4" fill="#1e293b" /><text x="92" y="57" text-anchor="middle" font-size="10" font-family="sans-serif" fill="white">upon</text>
      <rect x="120" y="42" width="20" height="22" rx="4" fill="#1e293b" /><text x="130" y="57" text-anchor="middle" font-size="10" font-family="sans-serif" fill="white">a</text>
      <rect x="146" y="42" width="40" height="22" rx="4" fill="#1e293b" /><text x="166" y="57" text-anchor="middle" font-size="10" font-family="sans-serif" fill="white">time</text>
      <text x="20" y="82" font-size="8" font-family="monospace" fill="#64748b">your story start</text>
      <path d="M192 53 L222 53" stroke="#d97706" stroke-width="2.4" />
      <polygon points="222,49 222,57 230,53" fill="#d97706" />
      <circle cx="192" cy="53" r="2.5" fill="#d97706" class="hero-flow" />

      <!-- scene 2: the model box -->
      <rect x="236" y="26" width="380" height="132" rx="8" fill="#f8fafc" stroke="#cbd5e1" />
      <text x="426" y="40" text-anchor="middle" font-size="8" font-family="monospace" font-weight="700" fill="#64748b">INSIDE THE MODEL · 6 LAYERS</text>

      <!-- attention arcs inside the box -->
      <circle cx="330" cy="58" r="3.5" fill="#475569" />
      <circle cx="390" cy="58" r="3.5" fill="#475569" />
      <circle cx="450" cy="58" r="3.5" fill="#475569" />
      <circle cx="510" cy="58" r="3.5" fill="#d97706" />
      <path d="M510 56 Q480 44 450 56" pathLength="100" class="hero-arc" fill="none" stroke="#d97706" stroke-width="1.6" opacity="0.9" />
      <path d="M510 56 Q450 38 390 56" pathLength="100" class="hero-arc" style="animation-delay: 160ms" fill="none" stroke="#d97706" stroke-width="1.2" opacity="0.6" />
      <path d="M510 56 Q420 32 330 56" pathLength="100" class="hero-arc" style="animation-delay: 320ms" fill="none" stroke="#d97706" stroke-width="0.9" opacity="0.4" />
      <text x="600" y="70" text-anchor="end" font-size="7.5" font-family="monospace" fill="#92400e">heads look back ↶</text>

      <!-- six layer bands -->
      {#each Array(6) as _, i}
        <rect x="252" y={72 + i * 13} width="348" height="10" rx="2" fill={i % 2 ? '#e9edf2' : '#eef1f5'} stroke="#d7dde5" stroke-width="0.5" />
        <text x="258" y={80 + i * 13} font-size="7" font-family="monospace" font-weight="700" fill="#64748b">L{i}</text>
      {/each}
      <text x="252" y="154" font-size="7.5" font-family="monospace" fill="#64748b">every layer is opened below in Act 1</text>

      <!-- arrow to output -->
      <path d="M616 92 L646 92" stroke="#d97706" stroke-width="2.4" />
      <polygon points="646,88 646,96 654,92" fill="#d97706" />
      <circle cx="616" cy="92" r="2.5" fill="#d97706" class="hero-flow hero-flow-delay" />

      <!-- output guess -->
      <rect x="656" y="76" width="84" height="28" rx="5" fill="#1e293b" />
      <text x="698" y="95" text-anchor="middle" font-size="11" font-family="sans-serif" font-weight="700" fill="white">" there"</text>
      <rect x="656" y="112" width="84" height="4" rx="2" fill="#e2e8f0" />
      <rect x="656" y="112" width="70" height="4" rx="2" fill="#d97706" />
      <text x="656" y="130" font-size="8" font-family="monospace" fill="#64748b">its next-word guess</text>

      <!-- divider -->
      <line x1="752" y1="28" x2="752" y2="158" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3" />

      <!-- scene 3: break it -->
      {#each Array(6) as _, i}
        <rect x={768 + i * 28} y="42" width="24" height="24" rx="4" fill={i === 3 ? '#b91c1c' : '#334155'} />
        <text x={780 + i * 28} y="58" text-anchor="middle" font-size="8" font-family="monospace" font-weight="700" fill="white">H{i}</text>
        {#if i === 3}
          <line x1={771 + i * 28} y1="45" x2={789 + i * 28} y2="63" stroke="white" stroke-width="2" />
          <line x1={789 + i * 28} y1="45" x2={771 + i * 28} y2="63" stroke="white" stroke-width="2" />
        {/if}
      {/each}
      <text x="768" y="82" font-size="8" font-family="monospace" fill="#64748b">switch one head off</text>

      <!-- before / after -->
      <rect x="768" y="98" width="92" height="26" rx="4" fill="white" stroke="#94a3b8" />
      <text x="814" y="115" text-anchor="middle" font-size="9" font-family="sans-serif" font-weight="700" fill="#1e293b">" and" 21%</text>
      <path d="M864 111 L878 111" stroke="#94a3b8" stroke-width="1.8" />
      <polygon points="878,107 878,115 886,111" fill="#94a3b8" />
      <rect x="888" y="98" width="92" height="26" rx="4" fill="#fff7ed" stroke="#c2410c" />
      <text x="934" y="115" text-anchor="middle" font-size="9" font-family="sans-serif" font-weight="700" fill="#9a3412">" with" 13%</text>
      <text x="768" y="142" font-size="8" font-family="monospace" fill="#64748b">the guess changes — that is what the head was doing</text>
    </svg>
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
