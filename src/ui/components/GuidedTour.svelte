<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{ close: void; done: void }>();

  export let open: boolean = false;

  interface Slide {
    id: number;
    title: string;
    caption: string;
    techNote: string;
  }

  const slides: Slide[] = [
    {
      id: 1,
      title: 'A word enters the model',
      caption: 'Each word is turned into a list of 288 numbers. These numbers are the model\'s starting "thought" about that word.',
      techNote: 'Technical: token embedding lookup produces a d=288 vector in the residual stream.',
    },
    {
      id: 2,
      title: 'Attention heads look back',
      caption: 'Each layer has 6 heads. Each head asks: "which earlier words matter here?" and adds their meaning into the current word\'s numbers.',
      techNote: 'Technical: multi-head self-attention computes softmax(QKᵀ/√d)·V for each head.',
    },
    {
      id: 3,
      title: 'Memory is mixed in',
      caption: 'After attention, each layer recalls learned patterns (grammar, story structure) and adds them into the working memory.',
      techNote: 'Technical: feed-forward network (SwiGLU FFN) expands d=288 → 768 → 288.',
    },
    {
      id: 4,
      title: 'A guess emerges',
      caption: 'After 6 layers, the final numbers are scored against all 32,000 vocabulary words. The highest score becomes the model\'s guess for the next word.',
      techNote: 'Technical: final RMSNorm + unembedding (tied weights) produces logits over vocab.',
    },
    {
      id: 5,
      title: 'You can turn a head off',
      caption: 'Switching one head off removes its 48 numbers from the working memory. Watch how the guess changes — that\'s what the head was contributing.',
      techNote: 'Technical: zero-ablation of head output before W₀ projection (causal intervention).',
    },
  ];

  let current = 0;
  let dismissed = false;

  function next() {
    if (current < slides.length - 1) current++;
    else done();
  }
  function prev() {
    if (current > 0) current--;
  }
  function done() {
    dismissed = true;
    dispatch('done');
    dispatch('close');
  }
  function close() {
    dismissed = true;
    dispatch('close');
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') close();
  }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if open && !dismissed}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-ink/60 backdrop-blur-sm p-4"
    on:click|self={close}
    role="presentation"
  >
    <div
      class="bg-white rounded-lg shadow-xl max-w-md w-full overflow-hidden"
      role="dialog"
      aria-modal="true"
      aria-label="Guided tour of the model"
    >
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-hairline">
        <div class="flex items-center gap-2">
          <span class="w-6 h-6 rounded-full bg-amber-100 text-amber-800 text-xs font-bold flex items-center justify-center font-mono">
            {current + 1}
          </span>
          <span class="text-[10px] font-mono font-bold uppercase tracking-wider text-ink-muted">
            of {slides.length}
          </span>
        </div>
        <button
          on:click={close}
          class="w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-100 active:scale-95 text-ink-secondary hover:text-ink transition"
          title="Close tour"
        >
          ✕
        </button>
      </div>

      <!-- Slide content -->
      <div class="px-4 py-5 min-h-[220px] flex flex-col">
        <h3 class="text-base font-bold text-ink font-serif mb-2">{slides[current].title}</h3>
        <p class="text-sm text-slate-800 font-medium leading-relaxed mb-3">{slides[current].caption}</p>

        <!-- SVG Diagram -->
        <div class="flex-1 flex items-center justify-center bg-slate-50 rounded border border-slate-200 p-3 mb-3">
          {#if current === 0}
            <!-- Word → Numbers -->
            <svg viewBox="0 0 320 80" class="w-full max-w-[280px]">
              <rect x="10" y="28" width="52" height="24" rx="4" fill="#1e293b" />
              <text x="36" y="44" text-anchor="middle" fill="white" font-size="10" font-family="monospace">"Once"</text>
              <path d="M62 40 L78 40" stroke="#d97706" stroke-width="2" />
              <polygon points="78,36 78,44 86,40" fill="#d97706" />
              {#each Array(8) as _, i}
                <rect
                  x={86 + i * 26}
                  y="22"
                  width="20"
                  height="36"
                  rx="2"
                  fill={i % 2 === 0 ? '#334155' : '#475569'}
                  opacity={0.3 + i * 0.08}
                />
              {/each}
              <text x="200" y="70" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">288 numbers</text>
            </svg>
          {:else if current === 1}
            <!-- Attention rays -->
            <svg viewBox="0 0 320 80" class="w-full max-w-[280px]">
              <rect x="10" y="28" width="40" height="24" rx="4" fill="#334155" />
              <text x="30" y="44" text-anchor="middle" fill="white" font-size="9" font-family="sans-serif">The</text>
              <rect x="60" y="28" width="52" height="24" rx="4" fill="#334155" />
              <text x="86" y="44" text-anchor="middle" fill="white" font-size="9" font-family="sans-serif">little</text>
              <rect x="122" y="28" width="42" height="24" rx="4" fill="#d97706" />
              <text x="143" y="44" text-anchor="middle" fill="white" font-size="9" font-family="sans-serif">girl</text>
              <path d="M143 40 C 143 15, 30 15, 30 28" fill="none" stroke="#d97706" stroke-width="2" opacity="0.7" />
              <path d="M143 40 C 143 20, 86 20, 86 28" fill="none" stroke="#d97706" stroke-width="1.5" opacity="0.4" />
              <text x="160" y="70" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">Head looks back to earlier words</text>
            </svg>
          {:else if current === 2}
            <!-- Memory mixing -->
            <svg viewBox="0 0 320 80" class="w-full max-w-[280px]">
              {#each Array(8) as _, i}
                <rect x={20 + i * 36} y="30" width="28" height="20" rx="2" fill="#475569" opacity={0.4 + i * 0.06} />
              {/each}
              <path d="M20 60 L300 60" stroke="#d97706" stroke-width="2" stroke-dasharray="4,3" />
              <text x="160" y="75" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">Learned patterns mixed into the numbers</text>
            </svg>
          {:else if current === 3}
            <!-- Scores → guess -->
            <svg viewBox="0 0 320 80" class="w-full max-w-[280px]">
              {#each Array(8) as _, i}
                <rect x={20 + i * 36} y="25" width="28" height="24" rx="2" fill="#475569" opacity={0.4 + i * 0.06} />
              {/each}
              <path d="M300 37 L310 37" stroke="#d97706" stroke-width="2" />
              <polygon points="310,33 310,41 318,37" fill="#d97706" />
              <rect x="200" y="55" width="80" height="20" rx="4" fill="#1e293b" />
              <text x="240" y="68" text-anchor="middle" fill="white" font-size="10" font-family="sans-serif">"with" (13%)</text>
              <text x="160" y="15" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">Scored against 32,000 words</text>
            </svg>
          {:else}
            <!-- Head ablation -->
            <svg viewBox="0 0 320 80" class="w-full max-w-[280px]">
              {#each Array(6) as _, i}
                <rect x={30 + i * 46} y="20" width="36" height="36" rx="4" fill={i === 1 ? '#b91c1c' : '#334155'} opacity={i === 1 ? 0.9 : 0.6} />
                <text x={48 + i * 46} y="42" text-anchor="middle" fill="white" font-size="9" font-family="monospace">H{i}</text>
              {/each}
              <line x1="66" y1="20" x2="102" y2="56" stroke="white" stroke-width="3" />
              <line x1="102" y1="20" x2="66" y2="56" stroke="white" stroke-width="3" />
              <text x="160" y="75" text-anchor="middle" fill="#64748b" font-size="8" font-family="sans-serif">One head's 48 numbers removed → guess changes</text>
            </svg>
          {/if}
        </div>

        <p class="text-[10px] font-mono text-ink-muted leading-snug">{slides[current].techNote}</p>
      </div>

      <!-- Footer nav -->
      <div class="flex items-center justify-between px-4 py-3 border-t border-hairline bg-canvas-elevated">
        <button
          on:click={prev}
          disabled={current === 0}
          class="px-3 py-1.5 text-xs font-bold text-slate-700 hover:text-ink disabled:opacity-30 disabled:cursor-not-allowed rounded hover:bg-slate-100 transition"
        >
          ← Back
        </button>
        <div class="flex gap-1">
          {#each slides as _, i}
            <span class={`w-1.5 h-1.5 rounded-full transition-colors ${i === current ? 'bg-amber-600' : 'bg-slate-300'}`}></span>
          {/each}
        </div>
        <button
          on:click={next}
          class="px-3 py-1.5 text-xs font-bold text-white bg-ink hover:bg-black rounded transition shadow-sm"
        >
          {current === slides.length - 1 ? 'Start exploring' : 'Next →'}
        </button>
      </div>
    </div>
  </div>
{/if}
