<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import type { ModelConfig, ModelWeights } from '../../engine/model';
  import type { Tokenizer } from '../../engine/tokenizer';
  import {
    compareHeadAblation,
    type AblationComparisonResult,
  } from '../../engine/analysis';
  import HeadGrid from './HeadGrid.svelte';
  import LogitWaterfall from './LogitWaterfall.svelte';

  export let config: ModelConfig | null = null;
  export let weights: ModelWeights | null = null;
  export let tokenizer: Tokenizer | null = null;
  export let currentActiveLayer: number = -1;
  // Measured per-head importance (‖Δlogits‖₂ on ablation), fills in progressively
  export let importance: (number | null)[][] | null = null;
  export let tokens: number[] = [1];
  export let activePos: number = 0;

  const dispatch = createEventDispatcher<{
    headHover: { layer: number; head: number | null };
  }>();

  let selectedLayer: number = 5;
  let selectedHead: number = 4;
  let ablatedLayer: number | null = 5;
  let ablatedHead: number | null = 4;
  let isComputing: boolean = false;
  let comparisonResult: AblationComparisonResult | null = null;

  let lastRanPos = -1;
  let lastRanTokenCount = 0;
  let lastRanLayer = -1;
  let lastRanHead = -1;

  function runAblation(layer = selectedLayer, head = selectedHead, pos = activePos, currentTokens = tokens) {
    if (!config || !weights || !tokenizer) return;
    isComputing = true;
    ablatedLayer = layer;
    ablatedHead = head;

    setTimeout(() => {
      try {
        comparisonResult = compareHeadAblation(
          currentTokens,
          pos,
          layer,
          head,
          config,
          weights,
          tokenizer
        );
      } catch (err) {
        console.error('Failed to run head ablation comparison:', err);
      } finally {
        isComputing = false;
      }
    }, 10);
  }

  // Reactively rerun when head is changed, active position changes, or sequence updates
  $: if (
    config &&
    weights &&
    tokenizer &&
    (activePos !== lastRanPos ||
      tokens.length !== lastRanTokenCount ||
      selectedLayer !== lastRanLayer ||
      selectedHead !== lastRanHead)
  ) {
    lastRanPos = activePos;
    lastRanTokenCount = tokens.length;
    lastRanLayer = selectedLayer;
    lastRanHead = selectedHead;
    runAblation(selectedLayer, selectedHead, activePos, tokens);
  }

  function handleHeadSelected(event: CustomEvent<{ layer: number; head: number }>) {
    selectedLayer = event.detail.layer;
    selectedHead = event.detail.head;
    runAblation(selectedLayer, selectedHead, activePos, tokens);
  }

  function handleHeadHover(event: CustomEvent<{ layer: number; head: number | null }>) {
    dispatch('headHover', event.detail);
  }
</script>

<div class="space-y-2.5">
  <!-- Interactive 6x6 Circuit Board Matrix with Hover Brushing -->
  <HeadGrid
    bind:selectedLayer
    bind:selectedHead
    {ablatedLayer}
    {ablatedHead}
    {currentActiveLayer}
    {importance}
    on:select={handleHeadSelected}
    on:hover={handleHeadHover}
  />

  <!-- Comparison Results Panel with Causal Storytelling Banner -->
  <div class="bg-white border border-hairline rounded p-3.5 shadow-subtle">
    <!-- Header with High-Contrast Typography -->
    <div class="flex items-center justify-between pb-2 mb-2 border-b border-hairline text-xs font-mono">
      <div class="flex items-center gap-2">
        <span class="text-ink font-bold text-xs uppercase tracking-wider">Causal Intervention</span>
        <span class="text-hairline-bright font-bold">/</span>
        <span class="text-ink-secondary font-sans font-bold">
          Head L{selectedLayer}:H{selectedHead} Switched Off
        </span>
      </div>

      <button
        on:click={() => runAblation(selectedLayer, selectedHead)}
        disabled={isComputing || !config || !weights || !tokenizer}
        class="px-3 py-1 bg-ink hover:bg-black active:scale-95 disabled:opacity-40 text-white rounded text-xs font-bold transition shadow-sm"
      >
        {#if isComputing}
          <span>Testing...</span>
        {:else}
          <span>Re-run test</span>
        {/if}
      </button>
    </div>

    <p class="text-xs font-sans font-medium text-slate-900 leading-relaxed mb-2">
      The experiment bench. Pick a head on the Circuit Map and this runs the model twice: once normally, once with that head's 48 channels zeroed. The numbers show exactly what the head was contributing.
    </p>

    {#if comparisonResult}
      <div class="space-y-2.5 transition-all duration-200">
        <!-- Causal Storytelling Banner (Explaining the Cause -> Effect) -->
        <div class={`rounded p-2 text-xs font-sans flex flex-col sm:flex-row sm:items-center justify-between gap-2 border ${
          comparisonResult.predictionFlipped
            ? 'bg-orange-50 border-accent-rust shadow-sm'
            : 'bg-canvas-elevated border-hairline'
        }`}>
          <div class="flex items-center gap-2">
            <span class={`w-1.5 h-1.5 rounded-full shrink-0 ${comparisonResult.predictionFlipped ? 'bg-accent-rust' : 'bg-slate-300'}`}></span>
            <span class="text-ink-secondary">
              {#if comparisonResult.predictionFlipped}
                Switching off head <strong>L{selectedLayer}:H{selectedHead}</strong> flips the top guess from
                <strong class="text-ink font-mono">"{comparisonResult.baselineTop.tokenPiece.trim()}"</strong> to
                <strong class="text-accent-rust font-mono">"{comparisonResult.ablatedTop.tokenPiece.trim()}"</strong>.
              {:else}
                The model still guesses <strong class="text-ink font-mono">"{comparisonResult.baselineTop.tokenPiece.trim()}"</strong>, but head <strong>L{selectedLayer}:H{selectedHead}</strong> shifts its score by
                <span class="font-mono text-ink font-semibold">
                  {comparisonResult.ablatedTop.delta !== undefined && comparisonResult.ablatedTop.delta >= 0 ? '+' : ''}{comparisonResult.ablatedTop.delta?.toFixed(2)}
                </span>
                (total shift Δ = {comparisonResult.l2Diff.toFixed(1)}).
              {/if}
            </span>
          </div>

          <span class="font-mono text-[10px] font-bold shrink-0 text-slate-800 bg-white px-2 py-0.5 rounded border border-slate-200">
            Tested on pos {comparisonResult.testedPos} "{comparisonResult.testedPiece.trim() || '<s>'}"
          </span>
        </div>

        <!-- Metrics Bar with High-Contrast Bold Text -->
        <div class="grid grid-cols-4 gap-2 text-center text-xs font-mono">
          <div
            class="bg-canvas-elevated border border-hairline rounded p-1.5"
            title="Total change across all 32,000 output scores when this head is switched off"
          >
            <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Total Shift Δ</div>
            <div class="text-base font-bold text-ink mt-0.5">
              {comparisonResult.l2Diff.toFixed(2)}
            </div>
          </div>

          <div
            class="bg-canvas-elevated border border-hairline rounded p-1.5"
            title="Largest change to any single token's score"
          >
            <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Biggest Hit</div>
            <div class="text-base font-bold text-accent-rust mt-0.5">
              {comparisonResult.maxAbsDiff.toFixed(2)}
            </div>
          </div>

          <div
            class="bg-canvas-elevated border border-hairline rounded p-1.5"
            title="Each head owns 48 of the 288 vector channels. Switching it off zeroes these numbers"
          >
            <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Channels Cut</div>
            <div class="text-xs font-bold text-ink-secondary mt-1">
              {comparisonResult.targetHead * 48}..{(comparisonResult.targetHead + 1) * 48 - 1}
            </div>
          </div>

          <div
            class="bg-canvas-elevated border border-hairline rounded p-1.5"
            title={`Tested on position ${comparisonResult.testedPos} ("${comparisonResult.testedPiece.trim() || '<s>'}") across the full sequence of words seen so far.`}
          >
            <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Tested On</div>
            <div class="text-xs font-bold text-ink-secondary mt-1 truncate">
              pos {comparisonResult.testedPos} ({comparisonResult.testedPiece.trim() || '<s>'})
            </div>
          </div>
        </div>

        <!-- Diverging Waterfall Delta Plot -->
        <LogitWaterfall
          ablatedTop5={comparisonResult.ablatedTop5}
          baselineTop5={comparisonResult.baselineTop5}
        />
      </div>
    {/if}
  </div>
</div>
