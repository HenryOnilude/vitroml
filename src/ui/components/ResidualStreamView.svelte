<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Tensor } from '../../engine/tensor';
  import type { StepState } from '../../engine/forward';
  import type { ModelConfig, ModelWeights } from '../../engine/model';
  import type { Tokenizer } from '../../engine/tokenizer';
  import { computeLogitLens, attributeStepWrites, type TokenPrediction } from '../../engine/analysis';

  const dispatch = createEventDispatcher<{
    selectHead: { layer: number; head: number };
    headFocus: { layer: number; head: number } | null;
  }>();

  export let stepState: StepState | null = null;
  export let highlightedHead: number | null = null;
  export let severedHead: number | null = null;
  export let config: ModelConfig | null = null;
  export let weights: ModelWeights | null = null;
  export let tokenizer: Tokenizer | null = null;
  export let currentToken: number = 1;
  export let focusMode: 'attention' | 'memory' | 'prediction' | null = null;

  interface BarData {
    index: number;
    headIndex: number;
    value: number;
    height: number;
    y: number;
    isPositive: boolean;
    color: string;
  }

  // One snapshot of the model's top hypothesis at a single forward yield.
  interface LensEntry {
    key: string;
    label: string;
    top: TokenPrediction[];
  }

  const CHART_WIDTH = 900;
  const CHART_HEIGHT = 120;
  const PADDING_TOP = 10;
  const PADDING_BOTTOM = 16;
  const DRAW_HEIGHT = CHART_HEIGHT - PADDING_TOP - PADDING_BOTTOM;
  const ZERO_Y = PADDING_TOP + DRAW_HEIGHT / 2;

  // Deterministic ghost silhouette: previews the 288-channel shape before data lands
  const ghostV = (i: number) =>
    Math.sin(i * 0.31) * Math.cos(i * 0.117) + Math.sin(i * 1.71) * 0.3;

  // Crameri `vik` diverging ramp: navy (negative) → ash (zero) → rust-ochre (positive)
  const VIK_STOPS: [number, [number, number, number]][] = [
    [-1.0, [0x00, 0x12, 0x61]],
    [-0.5, [0x30, 0x7d, 0xa6]],
    [0.0,  [0xec, 0xe5, 0xe0]],
    [0.5,  [0xc3, 0x72, 0x43]],
    [1.0,  [0x59, 0x00, 0x08]],
  ];

  function vikColor(t: number): string {
    const x = Math.max(-1, Math.min(1, t));
    for (let i = 0; i < VIK_STOPS.length - 1; i++) {
      const [t0, c0] = VIK_STOPS[i];
      const [t1, c1] = VIK_STOPS[i + 1];
      if (x >= t0 && x <= t1) {
        const f = (x - t0) / (t1 - t0);
        const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
      }
    }
    return 'rgb(89,0,8)';
  }

  let bars: BarData[] = [];
  let stats = { min: 0, max: 0, mean: 0, norm: 0, dim: 0 };
  let hoveredBar: BarData | null = null;
  let logitLensPredictions: TokenPrediction[] = [];
  let lensHistory: LensEntry[] = [];
  let lastRunToken: string | null = null;

  // Causal storytelling: Δx = what this phase actually WROTE into the residual stream
  let streamData: Float32Array | null = null;
  let deltaData: Float32Array | null = null;
  let prevStreamData: Float32Array | null = null;
  let lastStepKey: string | null = null;
  let viewMode: 'stream' | 'delta' = 'stream';

  $: if (stepState?.residual) {
    const tensor: Tensor = stepState.residual;
    const data = tensor.data.subarray(tensor.offset, tensor.offset + tensor.size);
    streamData = data;

    // Diff against the previous yielded snapshot to expose this phase's contribution.
    // Only meaningful within the same position; a new token starts a fresh stream.
    const stepKey = `${currentToken}:${stepState.position}:${stepState.layer}:${stepState.phase}`;
    if (stepKey !== lastStepKey) {
      if (prevStreamData && prevStreamData.length === data.length && lastStepKey?.startsWith(`${currentToken}:${stepState.position}:`)) {
        const d = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) d[i] = data[i] - prevStreamData[i];
        deltaData = d;
      } else {
        deltaData = null;
      }
      prevStreamData = Float32Array.from(data);
      lastStepKey = stepKey;
    }

    // Compute Logit Lens projection at this step
    if (config && weights && tokenizer) {
      try {
        logitLensPredictions = computeLogitLens(tensor, config, weights, tokenizer, currentToken, 3);

        // Record the evolving top hypothesis as the forward pass deepens.
        // History is per-position: a new token begins a fresh hypothesis chain.
        const runKey = `${currentToken}:${stepState.position}`;
        if (lastRunToken !== runKey) {
          lensHistory = [];
          lastRunToken = runKey;
        }
        const key = `${stepState.position}:${stepState.layer}-${stepState.phase}`;
        const entry: LensEntry = {
          key,
          label: `L${stepState.layer}${stepState.phase === 'post-attention' ? '·Attn' : '·FFN'}`,
          top: logitLensPredictions,
        };
        if (!lensHistory.some((e) => e.key === key)) {
          lensHistory = [...lensHistory, entry];
        } else {
          lensHistory = lensHistory.map((e) => (e.key === key ? entry : e));
        }
      } catch (err) {
        console.error('Logit lens error:', err);
      }
    }
  } else {
    streamData = null;
    deltaData = null;
    prevStreamData = null;
    lastStepKey = null;
    hoveredBar = null;
    logitLensPredictions = [];
    lensHistory = [];
    lastRunToken = null;
  }

  // Displayed quantity: absolute stream (x) or this phase's write (Δx)
  $: displayData = viewMode === 'delta' && deltaData ? deltaData : streamData;

  $: if (displayData) {
    const data = displayData;
    const len = data.length;
    stats.dim = len;

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sumSquares = 0;

    for (let i = 0; i < len; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      sumSquares += v * v;
    }

    stats.min = min;
    stats.max = max;
    stats.mean = sum / len;
    stats.norm = Math.sqrt(sumSquares);

    const maxAbs = Math.max(Math.abs(min), Math.abs(max), 0.001);
    const halfHeight = DRAW_HEIGHT / 2;

    bars = Array.from(data).map((v, i) => {
      const normalizedHeight = (Math.abs(v) / maxAbs) * halfHeight;
      const h = Math.max(1.5, normalizedHeight);
      const isPositive = v >= 0;
      const y = isPositive ? ZERO_Y - h : ZERO_Y;
      const headIndex = Math.floor(i / 48);
      return {
        index: i,
        headIndex,
        value: v,
        height: h,
        y,
        isPositive,
        color: vikColor(v / maxAbs),
      };
    });
  } else {
    bars = Array.from({ length: 288 }).map((_, i) => ({
      index: i,
      headIndex: Math.floor(i / 48),
      value: 0,
      height: 1.5,
      y: ZERO_Y - 0.75,
      isPositive: true,
      color: vikColor(0),
    }));
    stats = { min: 0, max: 0, mean: 0, norm: 0, dim: 288 };
  }

  // Latest snapshot and its immediate predecessor, for delta annotations
  $: latestLens = lensHistory.length > 0 ? lensHistory[lensHistory.length - 1] : null;
  $: prevLens = lensHistory.length > 1 ? lensHistory[lensHistory.length - 2] : null;

  function lensDeltaFor(tokenId: number): number | null {
    if (!prevLens || !latestLens) return null;
    const prev = prevLens.top.find((p) => p.tokenId === tokenId);
    const curr = latestLens.top.find((p) => p.tokenId === tokenId);
    if (!prev || !curr) return null;
    return curr.logit - prev.logit;
  }

  // "Why did the guess change?" — attribute this step's write across contributors
  $: why = (() => {
    if (!stepState?.writes || !config || !weights || !latestLens || latestLens.top.length === 0) {
      return null;
    }
    const target = latestLens.top[0];
    const attrs = attributeStepWrites(stepState.writes, stepState.residual, target.tokenId, weights, config.dim);
    const total = attrs.reduce((s, a) => s + a.effect, 0);
    const movers = [...attrs].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).slice(0, 2);
    return {
      phaseLabel: `L${stepState.writes.layer} ${stepState.writes.kind === 'heads' ? 'attention' : 'FFN'}`,
      guessPiece: target.tokenPiece.trim(),
      flipped: prevLens !== null && prevLens.top[0]?.tokenId !== target.tokenId,
      total,
      movers,
      isFfn: stepState.writes.kind === 'ffn',
    };
  })();
</script>

<div class={`bg-white border rounded p-3 shadow-subtle flex flex-col justify-between transition-all duration-300 ${focusMode === 'attention' ? 'border-hairline opacity-60 grayscale-[0.3]' : 'border-hairline'} ${focusMode === 'prediction' ? 'ring-2 ring-amber-400/60' : ''}`}>
  <!-- Header & Metric Badges with High-Contrast Typography -->
  <div class="flex flex-wrap items-center justify-between gap-y-1.5 gap-x-2 pb-2 mb-2 border-b border-hairline text-xs font-mono min-w-0">
    <div class="flex flex-wrap items-center gap-2 min-w-0">
      <span
        class="text-ink font-serif font-bold text-[15px] tracking-tight"
        title="Technical term: Working Vector / Residual Stream. The 288 numbers each layer reads from and writes back into."
      >Working Memory</span>
      <span class="text-slate-300 font-bold">/</span>
      {#if viewMode === 'delta' && stepState}
        <span class="text-ink-secondary font-sans font-semibold text-xs">
          Changes written by L{stepState.layer} {stepState.phase === 'post-attention' ? 'Attention' : 'Memory'}
        </span>
      {:else}
        <span class="text-ink-secondary font-sans font-semibold text-xs"><span class="font-semibold">Working Vector</span> · 288 Numbers</span>
      {/if}
      {#if stepState}
        <span class="text-amber-950 font-bold bg-amber-100/80 border border-amber-300 px-2.5 py-0.5 rounded text-[10px]">
          L{stepState.layer} · {stepState.phase === 'post-attention' ? 'Attention' : 'Memory'}
        </span>
      {/if}
    </div>

    <!-- Integrated Stats Summary with Bold Text -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs min-w-0">
      <!-- x / Δx view toggle -->
      <div class="flex items-center rounded border border-hairline overflow-hidden text-xs font-mono font-bold shadow-subtle">
        <button
          on:click={() => (viewMode = 'stream')}
          class={`px-3 py-1.5 sm:py-0.5 transition-colors active:scale-95 ${viewMode === 'stream' ? 'bg-ink text-white font-extrabold' : 'bg-white text-ink-secondary hover:text-ink'}`}
          title="The running vector as it currently stands"
        >
          x
        </button>
        <button
          on:click={() => (viewMode = 'delta')}
          disabled={!deltaData}
          class={`px-3 py-1.5 sm:py-0.5 transition-colors disabled:opacity-40 active:scale-95 ${viewMode === 'delta' ? 'bg-ink text-white font-extrabold' : 'bg-white text-ink-secondary hover:text-ink'}`}
          title="What this step added or removed from the vector"
        >
          Δx
        </button>
      </div>

      <div class="flex items-center gap-1 font-mono" title="Overall length of the vector">
        <span class="text-slate-600 font-bold text-[11px]">Size:</span>
        <span class="text-ink font-bold">{stats.norm.toFixed(2)}</span>
      </div>
      <div class="flex items-center gap-1 font-mono">
        <span class="text-slate-600 font-bold text-[11px]">Range:</span>
        <span class="text-slate-800 font-semibold">[{stats.min.toFixed(2)}, {stats.max.toFixed(2)}]</span>
      </div>
      <div class="flex items-center gap-1 font-mono">
        <span class="text-slate-600 font-bold text-[11px]">Avg:</span>
        <span class="text-slate-800 font-semibold">{stats.mean.toFixed(2)}</span>
      </div>
    </div>
  </div>

  <p class="text-xs font-sans font-medium text-slate-900 mb-1">
    The token's 288 working numbers, rewritten at every stage.
  </p>
  <details class="text-xs mb-2">
    <summary class="inline-flex items-center gap-1.5 cursor-pointer text-xs font-sans font-semibold text-slate-700 hover:text-ink hover:bg-slate-100 active:bg-slate-200 px-2 py-1 rounded transition select-none">
      <span class="w-3.5 h-3.5 rounded-full bg-slate-200 text-slate-700 text-[10px] flex items-center justify-center font-bold">?</span>
      <span>What's this?</span>
    </summary>
    <svg viewBox="0 0 320 48" class="w-full max-w-sm h-auto mt-1.5" aria-hidden="true">
      {#each Array(6) as _, i}
        <rect x={4 + i * 52} y="6" width="48" height="16" rx="3" fill={i % 2 ? '#e2e8f0' : '#cbd5e1'} stroke="#94a3b8" stroke-width="0.5" />
        <text x={28 + i * 52} y="17.5" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="700" fill="#334155">H{i}</text>
        <text x={28 + i * 52} y="32" text-anchor="middle" font-size="6.5" font-family="monospace" fill="#64748b">48 numbers</text>
      {/each}
      <text x="4" y="45" font-size="7" font-family="monospace" fill="#64748b">288 channels, grouped into the same six head slices as the Circuit Map</text>
    </svg>
    <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1.5 pl-1">
      The token's working notes. Every layer reads these 288 numbers and writes changes back into them. The shaded bands group the numbers into the same six head slices you see as Attention Field rows and as Circuit Map squares.
    </p>
  </details>

  <!-- Main Chart Canvas -->
  <div class="relative bg-slate-50 border border-slate-200 rounded p-2 overflow-hidden shadow-inner">
    <!-- SVG Bar Chart -->
    <svg
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      class="w-full h-[110px] overflow-visible"
      preserveAspectRatio="none"
    >
      <!-- Head Region Background Bands (6 Heads × 48 dims) -->
      {#each Array(6) as _, hIdx}
        {@const xStart = (hIdx * 48 / 288) * CHART_WIDTH}
        {@const xWidth = (48 / 288) * CHART_WIDTH}
        {@const isHeadHighlighted = highlightedHead === hIdx}
        {@const isHeadSevered = severedHead === hIdx}

        {#if isHeadSevered}
          <rect
            x={xStart}
            y={0}
            width={xWidth}
            height={CHART_HEIGHT}
            fill="#b91c1c"
            opacity="0.08"
          />
          <line
            x1={xStart}
            y1={1}
            x2={xStart + xWidth}
            y2={1}
            stroke="#b91c1c"
            stroke-width="2"
          />
        {:else if hIdx % 2 === 1}
          <rect
            x={xStart}
            y={0}
            width={xWidth}
            height={CHART_HEIGHT}
            fill="#000000"
            opacity="0.015"
          />
        {/if}

        {#if hIdx > 0}
          <line
            x1={xStart}
            y1={0}
            x2={xStart}
            y2={CHART_HEIGHT}
            stroke="#e2e8f0"
            stroke-width="1"
            stroke-dasharray="2,2"
          />
        {/if}

        {#if isHeadHighlighted}
          <rect
            x={xStart}
            y={0}
            width={xWidth}
            height={CHART_HEIGHT}
            fill="#09090b"
            opacity="0.08"
          />
        {/if}
      {/each}

      <!-- Zero baseline -->
      <line
        x1="0"
        y1={ZERO_Y}
        x2={CHART_WIDTH}
        y2={ZERO_Y}
        stroke="#cbd5e1"
        stroke-width="1"
      />

      <!-- Ghost bars: preview the 288-channel silhouette before the first pass writes real values -->
      {#if !stepState}
        {#each Array(288) as _, i}
          {@const v = ghostV(i)}
          {@const gh = Math.abs(v) * (DRAW_HEIGHT / 2) * 0.8}
          <rect
            x={i * (CHART_WIDTH / 288)}
            y={v >= 0 ? ZERO_Y - gh : ZERO_Y}
            width={Math.max(1.5, CHART_WIDTH / 288 - 0.35)}
            height={gh}
            fill="#94a3b8"
            opacity="0.22"
          />
        {/each}
      {/if}

      <!-- Bars: Precision Deep Blue (#2563eb) positive, Precision Rose (#e11d48) negative -->
      {#each bars as bar}
        {@const barWidth = CHART_WIDTH / bars.length}
        {@const isHeadFocused = highlightedHead !== null && bar.headIndex === highlightedHead}
        <rect
          x={bar.index * barWidth}
          y={bar.y}
          width={Math.max(1.5, barWidth - 0.35)}
          height={bar.height}
          fill={bar.color}
          opacity={
            hoveredBar && hoveredBar.index === bar.index
              ? 1.0
              : highlightedHead !== null && !isHeadFocused
              ? 0.18
              : hoveredBar && hoveredBar.index !== bar.index
              ? 0.35
              : 0.95
          }
          style="transition: height 180ms cubic-bezier(0.16, 1, 0.3, 1), y 180ms cubic-bezier(0.16, 1, 0.3, 1);"
          class="cursor-crosshair hover:opacity-100"
          role="graphics-symbol"
          aria-label={`Dim ${bar.index}: ${bar.value.toFixed(4)}`}
          on:mouseenter={() => (hoveredBar = bar)}
          on:mouseleave={() => (hoveredBar = null)}
        />
      {/each}
    </svg>

    <!-- Empty-state explainer: the stream is flat until a forward pass writes into it -->
    {#if !stepState}
      <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
        <span class="text-[11px] font-sans font-medium text-slate-900 bg-white/80 px-2 py-0.5 rounded">
          Faint bars preview the shape. Press Step or Play and real values land here, one layer at a time
        </span>
      </div>
    {/if}

    <!-- Hover Tooltip -->
    {#if hoveredBar}
      <div class="absolute top-2 left-2 bg-white border border-slate-300 rounded px-2.5 py-0.5 text-xs font-mono shadow-sm flex items-center gap-2">
        <span class="text-slate-800 font-bold">Channel [{hoveredBar.index}] · H{hoveredBar.headIndex}:</span>
        <span class="font-extrabold" style={`color: ${hoveredBar.color}`}>
          {hoveredBar.value.toFixed(4)}
        </span>
      </div>
    {/if}

    <!-- Head Region Footer Labels with High Contrast -->
    <div class="flex justify-between items-center text-[10px] text-slate-600 font-mono font-bold mt-0.5 px-1">
      {#each Array(6) as _, hIdx}
        <span
          class={`w-[16.6%] text-center transition-colors ${
            highlightedHead === hIdx
              ? 'text-ink font-black'
              : severedHead === hIdx
              ? 'text-accent-rust font-extrabold'
              : ''
          }`}
          title={`Head ${hIdx}'s slice of the vector: channels ${hIdx * 48} to ${hIdx * 48 + 47}${severedHead === hIdx ? ' · currently turned off in Act 2' : ''}`}
        >
          H{hIdx}{severedHead === hIdx ? ' ✕' : ''}
        </span>
      {/each}
    </div>
  </div>

  <!-- Hypothesis Evolution Tracker · Logit Lens -->
  <div class="mt-2.5 pt-2 border-t border-hairline text-[11px] font-mono min-w-0">
    <div class="flex flex-wrap items-center justify-between gap-y-1 gap-x-2 min-w-0">
      <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 min-w-0">
        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-800" title="Technical term: Logit Lens. Translates intermediate working memory into top vocabulary guesses at each layer.">Thought Evolution (Best Guess)</span>
        <span class="text-slate-300 font-bold">·</span>
        <span class="text-slate-600 font-sans font-medium">watch the model's guess mature across layers</span>
      </div>

      <div class="flex items-center gap-3 text-slate-800 font-bold">
        <span class="flex items-center gap-1.5">
          <span class="w-2.5 h-2.5 rounded shadow-sm" style="background: #c37243"></span>
          <span>above 0</span>
        </span>
        <span class="flex items-center gap-1.5">
          <span class="w-2.5 h-2.5 rounded shadow-sm" style="background: #307da6"></span>
          <span>below 0</span>
        </span>
      </div>
    </div>

    {#if lensHistory.length > 0}
      <!-- Depth progression: how the leading token candidate morphs L0 → L5 -->
      <div class="mt-2 flex items-center gap-1.5 overflow-x-auto scrollbar-none pb-1">
        {#each lensHistory as entry, i}
          {@const isLatest = i === lensHistory.length - 1}
          {@const top = entry.top[0]}
          {@const pct = top?.prob !== undefined ? Math.round(top.prob * 100) : null}
          {@const layerNum = parseInt(entry.label.replace(/^L(\d+).*/, '$1'), 10)}
          {@const displayLabel = Number.isFinite(layerNum) ? `L${layerNum}` : entry.label}
          {@const mappedHead = layerNum === 0 ? 1 : layerNum === 5 ? 3 : null}

          <div class="flex items-center gap-1.5 shrink-0">
            <div
              role="button"
              tabindex="0"
              on:click={() => mappedHead !== null && dispatch('selectHead', { layer: layerNum, head: mappedHead })}
              on:mouseenter={() => mappedHead !== null && dispatch('headFocus', { layer: layerNum, head: mappedHead })}
              on:mouseleave={() => dispatch('headFocus', null)}
              on:keydown={(e) => {
                if ((e.key === 'Enter' || e.key === ' ') && mappedHead !== null) {
                  dispatch('selectHead', { layer: layerNum, head: mappedHead });
                }
              }}
              class="flex flex-col items-center px-2 py-1 rounded border transition-all duration-200 {isLatest ? 'border-amber-500 bg-amber-50/80 shadow-sm ring-1 ring-amber-400' : 'border-slate-200 bg-slate-50'} {mappedHead !== null ? 'cursor-pointer hover:border-amber-400 hover:shadow-xs active:scale-95' : ''}"
              title={mappedHead !== null ? `Click to inspect L${layerNum}:H${mappedHead} attention rays · Top guess: "${top?.tokenPiece.trim() ?? ''}" (${pct ?? '?'}%)` : `${entry.label}: top guess "${top?.tokenPiece.trim() ?? ''}" (${pct ?? '?'}% confidence)`}
            >
              <span class="text-[8px] font-bold uppercase tracking-wider {isLatest ? 'text-amber-900 font-extrabold' : 'text-slate-500'}">{displayLabel}</span>
              <span class="text-xs font-bold font-sans whitespace-nowrap text-ink">{top?.tokenPiece ? `"${top.tokenPiece.trim()}"` : '·'}</span>
              {#if pct !== null}
                <span class="text-[9px] font-mono font-semibold {isLatest ? 'text-amber-800 font-bold' : 'text-slate-500'}">{pct}%</span>
              {/if}
            </div>
            {#if !isLatest}
              <span class="text-slate-400 font-bold text-xs select-none">→</span>
            {/if}
          </div>
        {/each}
      </div>

      <!-- Current step top-3 candidates with step-over-step logit deltas -->
      {#if latestLens}
        <div class="mt-1.5 flex items-center gap-1.5 overflow-x-auto scrollbar-none">
          <svg
            viewBox="0 0 58 22"
            class="w-14 h-[22px] shrink-0"
            role="img"
            title="How a guess appears: the 288 numbers are scored against all 32,000 vocabulary words, and the highest scores win"
          >
            <rect x="2" y="4" width="6" height="14" rx="1.5" fill="#334155" />
            {#each [2, 5.5, 9, 12.5, 16, 19.5] as y, i}
              <line x1="10" y1="11" x2="42" y2={y} stroke={i === 0 ? '#d97706' : '#cbd5e1'} stroke-width="1" />
              <rect x="43" y={y - 1} width={i === 0 ? 11 : 5} height="2" rx="1" fill={i === 0 ? '#d97706' : '#cbd5e1'} />
            {/each}
            <text x="44" y="21.5" font-size="5" font-family="monospace" fill="#94a3b8">32k</text>
          </svg>
          <span class="text-[10px] font-bold text-slate-600 uppercase tracking-wider shrink-0">Now:</span>
          {#each latestLens.top as pred}
            {@const delta = lensDeltaFor(pred.tokenId)}
            <span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-100 border border-slate-300 shrink-0">
              <span class="text-slate-900 font-bold font-sans">{pred.tokenPiece}</span>
              <span class="text-slate-700 font-mono text-[10px] font-bold">{pred.logit.toFixed(1)}</span>
              {#if delta !== null}
                <span class="text-[9px] font-bold {delta >= 0 ? 'text-emerald-700' : 'text-rose-700'}">
                  {delta >= 0 ? '▲' : '▼'}{Math.abs(delta).toFixed(1)}
                </span>
              {/if}
            </span>
          {/each}
        </div>
      {/if}

      <!-- Why did the guess change? — per-contributor attribution of this step's write -->
      {#if why}
        <p class="mt-1 text-[10px] font-sans font-medium text-slate-700 leading-snug">
          <span class="font-bold text-slate-900">Why?</span>
          <strong class="font-mono">{why.phaseLabel}</strong>
          {why.flipped ? 'flipped the guess to' : `pushed "${why.guessPiece}" ${why.total >= 0 ? 'up' : 'down'}`}
          {#if why.flipped}<strong class="font-mono">"{why.guessPiece}"</strong>{/if}
          ({why.total >= 0 ? '+' : ''}{why.total.toFixed(1)}).
          {#if !why.isFfn && why.movers.length > 0}
            Biggest push: {#each why.movers as m, i}{i > 0 ? ', ' : ''}<button
              type="button"
              on:click={() => dispatch('selectHead', { layer: m.layer, head: m.head })}
              class="font-mono font-bold text-accent-bronze underline decoration-dotted underline-offset-2 hover:text-ink transition-colors"
              title="Select this head in the Circuit Map"
            >head L{m.layer}:H{m.head}</button> ({m.effect >= 0 ? '+' : ''}{m.effect.toFixed(1)}){/each}.
          {:else if why.isFfn}
            Sole contributor: the layer's memory lookup.
          {/if}
          <span
            class="text-ink-muted cursor-help border-b border-dotted border-ink-muted"
            title="Estimated attribution: linearized through the final norm with its scale frozen. It suggests where the change came from; switching the head off tests it."
          >Estimated push</span>
        </p>
      {/if}
    {:else}
      <p class="mt-2 text-slate-800 font-medium leading-relaxed">Step the model forward. After each stage, this vector is scored against the vocabulary and the model's top guess appears here, evolving with depth</p>
    {/if}
  </div>
</div>
