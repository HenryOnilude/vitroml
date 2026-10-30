<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  const LAYERS = [0, 1, 2, 3, 4, 5];

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

  // Mini residual-stream glyph: 14 bars, mixed signs
  const RESIDUAL_BARS = [
    0.55, -0.3, 0.8, -0.65, 0.4, 0.95, -0.45, 0.3,
    -0.85, 0.6, -0.25, 0.75, -0.5, 0.35,
  ];

  // Mini attention-field glyph: 4 rows × 8 cols, deterministic weights
  const ATTN_ROWS = [
    [0.9, 0.15, 0.4, 0.1, 0.05, 0.2, 0.1, 0.05],
    [0.1, 0.85, 0.2, 0.3, 0.1, 0.05, 0.15, 0.1],
    [0.2, 0.1, 0.7, 0.9, 0.3, 0.2, 0.1, 0.05],
    [0.05, 0.2, 0.1, 0.3, 0.8, 0.95, 0.4, 0.6],
  ];

  // Mini circuit-map glyph: 6×6 ink-wash saliency with one rust "switched off" cell
  const CIRCUIT_CELLS = Array.from({ length: 36 }, (_, i) =>
    i === 23 ? -1 : 0.06 + ((i * 37) % 10) / 60
  );

  // cividis ramp for the attention glyph (matches AttentionMap)
  const CIVIDIS_STOPS: [number, [number, number, number]][] = [
    [0.0,  [0x00, 0x22, 0x4e]],
    [0.25, [0x43, 0x4e, 0x6c]],
    [0.5,  [0x7d, 0x7c, 0x78]],
    [0.75, [0xbc, 0xae, 0x6c]],
    [1.0,  [0xfe, 0xe8, 0x38]],
  ];
  function cividis(t: number): string {
    const x = Math.max(0, Math.min(1, t));
    for (let i = 0; i < CIVIDIS_STOPS.length - 1; i++) {
      const [t0, c0] = CIVIDIS_STOPS[i];
      const [t1, c1] = CIVIDIS_STOPS[i + 1];
      if (x >= t0 && x <= t1) {
        const f = (x - t0) / (t1 - t0);
        const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
      }
    }
    return 'rgb(254,232,56)';
  }
</script>

<section class="bg-white border border-hairline rounded shadow-subtle px-4 sm:px-5 py-4">
  <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
    <!-- Left: headline + copy + specs -->
    <div class="max-w-xl">
      <div class="text-[9px] font-mono font-bold uppercase tracking-[0.2em] text-accent-bronze mb-1">
        Interactive explorable · client-side
      </div>
      <h1 class="text-lg sm:text-2xl font-bold tracking-tight text-ink font-serif">
        Watch a tiny language model think
      </h1>
      <p class="text-xs text-slate-800 font-medium leading-relaxed mt-1.5 font-sans">
        stories15M is a 15-million-parameter transformer small enough to run entirely in your browser.
        Step through it layer by layer, watch it write a story word by word,
        and switch off attention heads to see how the answer changes.
      </p>
      <div class="flex flex-wrap items-center gap-2 mt-2.5 text-[10px] font-mono text-ink-muted font-bold">
        <span>15.2M params</span>
        <span class="text-hairline-bright">·</span>
        <span>6 layers × 6 heads</span>
        <span class="text-hairline-bright">·</span>
        <span>288 channels</span>
        <span class="text-hairline-bright">·</span>
        <span>32k words</span>
        <span class="text-hairline-bright">·</span>
        <span>runs locally, no server</span>
      </div>
    </div>

    <!-- Right: capabilities as panel thumbnails -->
    <div class="shrink-0 flex flex-col gap-2.5 text-[11px] font-sans font-medium text-ink-secondary min-w-0">
      <!-- Step through the forward pass → residual stream bars -->
      <span class="flex items-center gap-2">
        <span class="w-8 h-5 flex items-end gap-[1.5px] shrink-0" aria-hidden="true">
          {#each RESIDUAL_BARS as v}
            <span
              class="w-[2px]"
              style={`height: ${Math.abs(v) * 100}%; background: ${v >= 0 ? '#c37243' : '#307da6'}`}
            ></span>
          {/each}
        </span>
        Step through the forward pass
      </span>

      <!-- See where each head looks → attention field -->
      <span class="flex items-center gap-2">
        <span class="w-8 h-5 flex flex-col gap-[1.5px] shrink-0" aria-hidden="true">
          {#each ATTN_ROWS as row}
            <span class="flex-1 flex gap-[1.5px]">
              {#each row as w}
                <span class="flex-1 rounded-[0.5px]" style={`background: ${cividis(w)}`}></span>
              {/each}
            </span>
          {/each}
        </span>
        See where each head looks
      </span>

      <!-- Switch heads off → circuit map with one hot cell -->
      <span class="flex items-center gap-2">
        <span class="w-8 h-5 grid grid-cols-6 gap-[1px] shrink-0" aria-hidden="true">
          {#each CIRCUIT_CELLS as w}
            {#if w < 0}
              <span class="rounded-[0.5px]" style="background: #b91c1c; background-image: repeating-linear-gradient(45deg, rgba(255,255,255,0.3) 0 1px, transparent 1px 2px)"></span>
            {:else}
              <span class="rounded-[0.5px]" style={`background: rgba(17,18,21,${w})`}></span>
            {/if}
          {/each}
        </span>
        Switch heads off, measure the change
      </span>
    </div>
  </div>

  <!-- Animated mini-pipeline: previews the real execution bus below -->
  <div class="relative mt-3.5 pt-3 border-t border-hairline">
    <div class="relative flex items-center justify-between gap-1.5" aria-hidden="true">
      <div class="absolute left-3 right-3 top-1/2 -translate-y-1/2 h-[1px] bg-slate-200"></div>

      <!-- traveling packet -->
      <span class="hero-packet absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-amber-500 shadow-[0_0_6px_2px_rgba(217,119,6,0.4)] z-10"></span>

      <span class="relative z-10 bg-ink text-white text-[9px] font-mono font-bold px-1.5 py-0.5 rounded">"Once"</span>
      <span class="relative z-10 bg-white border border-slate-300 text-ink-secondary text-[9px] font-mono font-bold px-1.5 py-0.5 rounded">E</span>
      {#each LAYERS as l}
        <span class="relative z-10 bg-white border border-slate-300 text-ink-secondary text-[9px] font-mono font-bold px-1.5 py-0.5 rounded">L{l}</span>
      {/each}
      <span class="relative z-10 bg-white border border-slate-300 text-ink-secondary text-[9px] font-mono font-bold px-1.5 py-0.5 rounded">N</span>
      <span class="relative z-10 bg-ink text-white text-[9px] font-mono font-bold px-1.5 py-0.5 rounded">scores</span>
    </div>

    <!-- The payoff: the model writing a story, token by token -->
    <div class="mt-2 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] font-mono min-w-0" aria-hidden="true">
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
          First time? Press <strong>▶ Play</strong> below. The model starts writing and everything lights up.
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
  .hero-packet {
    animation: hero-travel 6s cubic-bezier(0.45, 0, 0.55, 1) infinite;
  }
  @keyframes hero-travel {
    0%   { left: 0.75rem; opacity: 0; }
    6%   { opacity: 1; }
    94%  { opacity: 1; }
    100% { left: calc(100% - 0.75rem); opacity: 0; }
  }
  .hero-caret {
    animation: hero-blink 1.1s steps(2, start) infinite;
  }
  @keyframes hero-blink {
    50% { opacity: 0; }
  }
</style>
