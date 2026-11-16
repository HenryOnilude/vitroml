<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import type { ModelConfig, ModelWeights } from '../../engine/model';
  import type { Tokenizer } from '../../engine/tokenizer';
  import {
    compareHeadAblation,
    type AblationComparisonResult,
  } from '../../engine/analysis';
  import HeadGrid from './HeadGrid.svelte';

  export let config: ModelConfig | null = null;
  export let weights: ModelWeights | null = null;
  export let tokenizer: Tokenizer | null = null;
  export let currentActiveLayer: number = -1;
  // Measured per-head importance (‖Δlogits‖₂ on ablation), fills in progressively
  export let importance: (number | null)[][] | null = null;
  export let tokens: number[] = [1];
  export let activePos: number = 0;
  export let highlightedTarget: 'controls' | 'circuit' | 'result' | null = null;

  export function selectHead(layer: number, head: number) {
    selectedLayer = layer;
    selectedHead = head;
    runAblation(layer, head, activePos, tokens);
  }

  const dispatch = createEventDispatcher<{
    headHover: { layer: number; head: number | null };
    headSelect: { layer: number; head: number };
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
    dispatch('headSelect', event.detail);
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
    highlighted={highlightedTarget === 'circuit'}
    on:select={handleHeadSelected}
    on:hover={handleHeadHover}
  />

  <!-- Comparison Results Panel with Causal Storytelling Banner -->
  <div
    id="panel-experiment-result"
    class={`bg-white border rounded p-3.5 shadow-subtle transition-all duration-300 ${
      highlightedTarget === 'result' ? 'border-amber-500 ring-2 ring-amber-400/70' : 'border-hairline'
    }`}
  >
    <!-- Header with High-Contrast Typography -->
    <div class="flex flex-wrap items-center justify-between gap-y-1.5 gap-x-2 pb-2 mb-2 border-b border-hairline text-xs font-mono min-w-0">
      <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 min-w-0">
        <span class="text-ink font-bold text-xs uppercase tracking-wider" title="Causal intervention: see what changes when a specific attention head is turned off.">Turn Off a Head</span>
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

    <p class="text-xs font-sans font-medium text-slate-900 mb-1">
      The experiment bench. Cut a head, measure the change.
    </p>
    <details class="text-xs mb-2">
      <summary class="inline-flex items-center gap-1.5 cursor-pointer text-xs font-sans font-semibold text-slate-700 hover:text-ink hover:bg-slate-100 active:bg-slate-200 px-2 py-1 rounded transition select-none">
        <span class="w-3.5 h-3.5 rounded-full bg-slate-200 text-slate-700 text-[10px] flex items-center justify-center font-bold">?</span>
        <span>What's this?</span>
      </summary>
      <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1.5 pl-1">
        The experiment bench. Pick a head on the Circuit Map and this runs the model twice: once normally, once with that head's 48 channels zeroed. The numbers show exactly what the head was contributing.
      </p>
    </details>

    {#if comparisonResult}
      {@const maxProb = Math.max(...comparisonResult.top3Pairs.flatMap((p) => [p.baselineProb, p.ablatedProb]), 0.01)}
      {@const beforePiece = comparisonResult.baselineTop.tokenPiece.trim().slice(0, 8)}
      {@const afterPiece = comparisonResult.ablatedTop.tokenPiece.trim().slice(0, 8)}
      <div class="space-y-2.5 transition-all duration-200">
        <!-- Experiment diagram: which head was cut, and what came out before vs after -->
        <svg
          viewBox="0 0 340 44"
          class="w-full h-11"
          role="img"
          aria-label={`Head ${selectedHead} of layer ${selectedLayer} was switched off; the guess changed from "${beforePiece}" to "${afterPiece}"`}
        >
          <!-- six heads of the layer, target crossed out -->
          {#each Array(6) as _, i}
            <rect
              x={4 + i * 24}
              y="8"
              width="20"
              height="20"
              rx="3"
              fill={i === comparisonResult.targetHead ? '#b91c1c' : '#334155'}
              opacity={i === comparisonResult.targetHead ? 0.95 : 0.55}
            />
            <text x={14 + i * 24} y="22" text-anchor="middle" font-size="8" font-family="monospace" font-weight="700" fill="white">H{i}</text>
            {#if i === comparisonResult.targetHead}
              <line x1={8 + i * 24} y1="11" x2={20 + i * 24} y2="25" stroke="white" stroke-width="1.6" />
              <line x1={20 + i * 24} y1="11" x2={8 + i * 24} y2="25" stroke="white" stroke-width="1.6" />
            {/if}
          {/each}
          <text x="4" y="40" font-size="6.5" font-family="monospace" fill="#64748b">L{selectedLayer} · H{comparisonResult.targetHead} off</text>

          <!-- arrow to before -->
          <line x1="152" y1="18" x2="168" y2="18" stroke="#94a3b8" stroke-width="1.5" />
          <polygon points="168,15 168,21 175,18" fill="#94a3b8" />

          <!-- before chip -->
          <rect x="176" y="6" width="66" height="26" rx="4" fill="white" stroke="#94a3b8" />
          <text x="209" y="17" text-anchor="middle" font-size="9" font-family="sans-serif" font-weight="700" fill="#1e293b">"{beforePiece}"</text>
          <text x="209" y="27" text-anchor="middle" font-size="7" font-family="monospace" fill="#64748b">{(comparisonResult.baselineTopProb * 100).toFixed(0)}% before</text>

          <!-- arrow to after -->
          <line x1="246" y1="18" x2="262" y2="18" stroke="#94a3b8" stroke-width="1.5" />
          <polygon points="262,15 262,21 269,18" fill="#94a3b8" />

          <!-- after chip -->
          <rect x="270" y="6" width="66" height="26" rx="4" fill={comparisonResult.predictionFlipped ? '#fff7ed' : 'white'} stroke={comparisonResult.predictionFlipped ? '#c2410c' : '#94a3b8'} />
          <text x="303" y="17" text-anchor="middle" font-size="9" font-family="sans-serif" font-weight="700" fill={comparisonResult.predictionFlipped ? '#9a3412' : '#1e293b'}>"{afterPiece}"</text>
          <text x="303" y="27" text-anchor="middle" font-size="7" font-family="monospace" fill="#64748b">{(comparisonResult.ablatedTopProb * 100).toFixed(0)}% after</text>
        </svg>

        <!-- Plain-English verdict -->
        <div class={`rounded p-2 text-xs font-sans border ${
          comparisonResult.predictionFlipped
            ? 'bg-orange-50 border-accent-rust shadow-sm'
            : 'bg-canvas-elevated border-hairline'
        }`}>
          <span class={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 align-middle ${comparisonResult.predictionFlipped ? 'bg-accent-rust' : 'bg-slate-300'}`}></span>
          <span class="text-ink">
            With <strong class="font-mono">L{selectedLayer}:H{selectedHead}</strong> off, the model
            {#if comparisonResult.predictionFlipped}
              says <strong class="font-mono">"{comparisonResult.ablatedTop.tokenPiece.trim()}"</strong>
              instead of <strong class="font-mono">"{comparisonResult.baselineTop.tokenPiece.trim()}"</strong>.
            {:else if comparisonResult.ablatedTopProb < comparisonResult.baselineTopProb}
              still says <strong class="font-mono">"{comparisonResult.baselineTop.tokenPiece.trim()}"</strong>,
              but its confidence dropped from {(comparisonResult.baselineTopProb * 100).toFixed(0)}%
              to {(comparisonResult.ablatedTopProb * 100).toFixed(0)}%.
            {:else}
              still says <strong class="font-mono">"{comparisonResult.baselineTop.tokenPiece.trim()}"</strong>,
              and its confidence barely moved
              ({(comparisonResult.baselineTopProb * 100).toFixed(0)}% → {(comparisonResult.ablatedTopProb * 100).toFixed(0)}%).
            {/if}
          </span>
        </div>

        <!-- Paired before/after bars: baseline top-3, same order and scale -->
        <div class="space-y-1.5">
          <div class="flex items-center gap-3 text-[9px] font-mono font-bold uppercase tracking-wider text-ink-muted">
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-[1px] bg-slate-400"></span>before</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-[1px] bg-amber-600"></span>after</span>
          </div>
          {#each comparisonResult.top3Pairs as pair}
            <div class="space-y-0.5">
              <div class="flex items-center justify-between text-[11px]">
                <span class="font-sans font-bold text-ink">"{pair.tokenPiece.trim()}"</span>
                <span class="font-mono text-[10px] text-ink-muted">{(pair.baselineProb * 100).toFixed(0)}% → {(pair.ablatedProb * 100).toFixed(0)}%</span>
              </div>
              <div class="h-1.5 bg-canvas-elevated rounded-full overflow-hidden">
                <div class="h-full bg-slate-400 rounded-full" style={`width: ${(pair.baselineProb / maxProb) * 100}%`}></div>
              </div>
              <div class="h-1.5 bg-canvas-elevated rounded-full overflow-hidden">
                <div class="h-full bg-amber-600 rounded-full" style={`width: ${(pair.ablatedProb / maxProb) * 100}%`}></div>
              </div>
            </div>
          {/each}
        </div>

        <!-- Technical metrics, collapsed -->
        <details class="text-xs">
          <summary class="cursor-pointer text-[10px] font-sans font-semibold text-ink-muted hover:text-ink select-none">Details</summary>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs font-mono mt-1.5">
            <div
              class="bg-canvas-elevated border border-hairline rounded p-1.5"
              title="Total change across all 32,000 output scores when this head is switched off (L2 difference)"
            >
              <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Total Shift</div>
              <div class="text-base font-bold text-ink mt-0.5">
                {comparisonResult.l2Diff.toFixed(2)}
              </div>
            </div>

            <div
              class="bg-canvas-elevated border border-hairline rounded p-1.5"
              title="Largest change to any single token's score"
            >
              <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Largest Shift</div>
              <div class="text-base font-bold text-accent-rust mt-0.5">
                {comparisonResult.maxAbsDiff.toFixed(2)}
              </div>
            </div>

            <div
              class="bg-canvas-elevated border border-hairline rounded p-1.5"
              title="Each head controls 48 of the 288 working memory numbers. Switching it off sets these to 0"
            >
              <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Signals Cut</div>
              <div class="text-xs font-bold text-ink-secondary mt-1">
                {comparisonResult.targetHead * 48}..{(comparisonResult.targetHead + 1) * 48 - 1}
              </div>
            </div>

            <div
              class="bg-canvas-elevated border border-hairline rounded p-1.5"
              title={`Tested on position ${comparisonResult.testedPos} ("${comparisonResult.testedPiece.trim() || '<s>'}") across the full sequence of words seen so far.`}
            >
              <div class="text-[10px] font-bold text-ink-muted uppercase tracking-wider">Tested Word</div>
              <div class="text-xs font-bold text-ink-secondary mt-1 truncate">
                pos {comparisonResult.testedPos} ({comparisonResult.testedPiece.trim() || '<s>'})
              </div>
            </div>
          </div>
        </details>
      </div>
    {/if}
  </div>
</div>
