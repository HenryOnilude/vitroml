<script lang="ts">
  import { fly } from 'svelte/transition';
  import type { ExecutionState } from '../../bridge/controller';

  export let state: ExecutionState | null = null;
  export let currentToken: number = 1;
  export let currentPos: number = 0;
  export let sampledPiece: string | null = null;

  // Dynamic narrative generation based on exact transformer forward-pass state
  interface NarrativeStep {
    phaseTag: string;
    headline: string;
    explanation: string;
  }

  function getNarrative(state: ExecutionState | null): NarrativeStep {
    if (!state || state.status === 'idle') {
      return {
        phaseTag: '01 / INITIALIZE',
        headline: 'Token Ready at the Input',
        explanation: `The next token waits at the model's input. Press "Step" or "Play" to watch it travel through all 6 layers.`,
      };
    }

    if (state.status === 'completed') {
      return {
        phaseTag: '03 / PREDICTION',
        headline: sampledPiece
          ? `Model Chose "${sampledPiece}". Feeding It Back In`
          : 'Scoring Every Possible Next Token',
        explanation: sampledPiece
          ? `"${sampledPiece}" had the highest score, so it becomes the next input at position ${currentPos + 1}. Press "Next token" or "Continue" to run it. The model still remembers everything before.`
          : 'The final vector is scored against all 32,000 vocabulary tokens. The highest score wins. Press "Next token" to feed the winner back as input.',
      };
    }

    if (!state.currentStep) {
      return {
        phaseTag: 'QUEUED',
        headline: `Token ${currentToken} Queued at Position ${currentPos}`,
        explanation: 'The model remembers every earlier token, so each attention head can look back across the full sequence. Step to push this token through the 6 layers.',
      };
    }

    const { layer, phase } = state.currentStep;

    if (phase === 'post-attention') {
      if (layer === 0) {
        return {
          phaseTag: `LAYER ${layer} · ATTENTION`,
          headline: 'Heads Gather Nearby Context',
          explanation: `Each of the 6 heads asks "which earlier tokens matter here?" and adds its answer into the stream (x ← x + attention).`,
        };
      } else if (layer < 4) {
        return {
          phaseTag: `LAYER ${layer} · ATTENTION`,
          headline: 'Heads Combine Earlier Words',
          explanation: `Heads compare this token against everything before it, pulling together meaning and grammar cues from the sequence.`,
        };
      } else {
        return {
          phaseTag: `LAYER ${layer} · ATTENTION`,
          headline: 'Heads Steer Toward an Answer',
          explanation: `The last attention pass nudges the stream toward real vocabulary. Switching heads off here moves the final guess the most.`,
        };
      }
    } else {
      // post-ffn
      if (layer === 0) {
        return {
          phaseTag: `LAYER ${layer} · FFN`,
          headline: 'First Memory Lookup',
          explanation: `The vector expands 288 → 768 through a gated filter (SwiGLU). This is where the model starts mixing in patterns it learned during training.`,
        };
      } else if (layer < 4) {
        return {
          phaseTag: `LAYER ${layer} · FFN`,
          headline: 'Memory Lookup',
          explanation: `These neurons work like an associative memory. They inject story patterns learned from TinyStories into the stream.`,
        };
      } else {
        return {
          phaseTag: `LAYER ${layer} · FFN`,
          headline: 'Last Refinement Pass',
          explanation: `The final processing step. The stream now carries the model's settled guess about which word comes next.`,
        };
      }
    }
  }

  // Progressive disclosure: plain explanation first, exact math on demand
  const MATH_NOTES: Record<string, string> = {
    'post-attention': 'softmax(Q·Kᵀ / √d) mixes values from earlier positions → each head writes a 48-d output → summed into the stream: x ← x + Attn(x)',
    'post-ffn': 'SwiGLU(x) = (xW₁ ⊙ SiLU(xW₃))W₂. The vector expands 288 → 768 → 288, then x ← x + FFN(x)',
  };

  let showMath = false;
  $: mathNote = state?.currentStep ? MATH_NOTES[state.currentStep.phase] : null;

  $: narrative = getNarrative(state);

  // Narrative arc: 01 Query → 02 Latent Journey → 03 Prediction (martini-glass structure)
  const CHAPTERS = ['Input', 'Through Layers', 'Prediction'];
  const TOTAL_YIELDS = 12;

  $: chapter = !state || state.status === 'idle' ? 0 : state.status === 'completed' ? 2 : 1;
  $: yieldIndex = state?.currentStep
    ? state.currentStep.layer * 2 + (state.currentStep.phase === 'post-ffn' ? 2 : 1)
    : state?.status === 'completed'
    ? TOTAL_YIELDS
    : 0;
  $: progressPct = Math.min(100, (yieldIndex / TOTAL_YIELDS) * 100);
</script>

<div class="bg-white border border-hairline rounded p-3 shadow-subtle flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 transition-all duration-200">
  <!-- Left Narrative Text -->
  <div class="flex items-start gap-3">
    <div class="w-2 h-2 rounded-full bg-accent-bronze mt-1 shrink-0 animate-pulse"></div>

    <div class="space-y-0.5">
      {#key narrative.headline}
        <div class="space-y-0.5" in:fly={{ y: 5, duration: 220 }}>
          <div class="flex items-center gap-2">
            <span class="text-[10px] font-mono font-bold uppercase tracking-wider text-accent-bronze">
              {narrative.phaseTag}
            </span>
            <span class="text-slate-300 font-bold">•</span>
            <span class="text-sm font-bold text-ink tracking-tight font-serif">
              {narrative.headline}
            </span>
          </div>

          <p class="text-xs text-ink leading-relaxed font-sans font-medium max-w-4xl">
            {narrative.explanation}
            {#if mathNote}
              <button
                on:click={() => (showMath = !showMath)}
                class="ml-1.5 text-[9px] font-mono font-bold uppercase tracking-wider text-accent-bronze hover:text-ink border border-amber-300 hover:border-ink rounded px-1 py-px align-middle transition-colors"
              >
                {showMath ? 'hide math' : 'math ▸'}
              </button>
            {/if}
          </p>

          {#if showMath && mathNote}
            <p class="text-[10px] font-mono text-ink-secondary leading-relaxed max-w-4xl pt-0.5 border-t border-dashed border-hairline mt-0.5">
              {mathNote}
            </p>
          {/if}
        </div>
      {/key}
    </div>
  </div>

  <!-- Right: Chapter arc + forward-pass progress -->
  <div class="shrink-0 flex flex-col items-end gap-1.5 self-stretch justify-center min-w-[190px]">
    <div class="flex items-center gap-1 text-[9px] font-mono font-bold uppercase tracking-wider">
      {#each CHAPTERS as ch, i}
        <span class={`transition-colors duration-200 ${
          i === chapter ? 'text-ink' : i < chapter ? 'text-ink-secondary' : 'text-ink-muted'
        }`}>
          {String(i + 1).padStart(2, '0')} {ch}
        </span>
        {#if i < CHAPTERS.length - 1}
          <span class="text-hairline-bright mx-0.5">→</span>
        {/if}
      {/each}
    </div>

    <div class="w-full">
      <div class="flex items-center justify-between text-[9px] font-mono font-bold text-ink-muted mb-0.5">
        <span>FORWARD PASS</span>
        <span class="text-ink-secondary">{yieldIndex} / {TOTAL_YIELDS}</span>
      </div>
      <div class="h-1 w-full bg-canvas-elevated border border-hairline rounded-full overflow-hidden">
        <div
          class="h-full bg-accent-bronze transition-all duration-300 ease-out"
          style={`width: ${progressPct}%`}
        ></div>
      </div>
    </div>
  </div>
</div>
