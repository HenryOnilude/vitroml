<script lang="ts">
  import type { Tensor } from '../../engine/tensor';
  import type { StepState } from '../../engine/forward';
  import type { ModelConfig, ModelWeights } from '../../engine/model';
  import type { Tokenizer } from '../../engine/tokenizer';
  import { computeLogitLens, type TokenPrediction } from '../../engine/analysis';

  export let stepState: StepState | null = null;
  export let highlightedHead: number | null = null;
  export let config: ModelConfig | null = null;
  export let weights: ModelWeights | null = null;
  export let tokenizer: Tokenizer | null = null;
  export let currentToken: number = 1;

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
</script>

<div class="bg-white border border-hairline rounded p-3 shadow-subtle flex flex-col justify-between">
  <!-- Header & Metric Badges with High-Contrast Typography -->
  <div class="flex flex-wrap items-center justify-between gap-y-1.5 gap-x-2 pb-2 mb-2 border-b border-hairline text-xs font-mono min-w-0">
    <div class="flex flex-wrap items-center gap-2 min-w-0">
      <span
        class="text-ink font-bold text-xs uppercase tracking-wider"
        title="The vector each token carries through the model. Every layer reads it and writes back into it. Also called the residual stream."
      >Working Vector</span>
      <span class="text-slate-300 font-bold">/</span>
      {#if viewMode === 'delta' && stepState}
        <span class="text-ink-secondary font-sans font-semibold text-xs">
          Δx written by L{stepState.layer} {stepState.phase === 'post-attention' ? 'Attention' : 'FFN'}
        </span>
      {:else}
        <span class="text-ink-secondary font-sans font-semibold text-xs">288 Dimensions</span>
      {/if}
      {#if stepState}
        <span class="text-amber-950 font-bold bg-amber-100/80 border border-amber-300 px-2.5 py-0.5 rounded text-[10px]">
          L{stepState.layer} · {stepState.phase === 'post-attention' ? 'Attn' : 'FFN'}
        </span>
      {/if}
    </div>

    <!-- Integrated Stats Summary with Bold Text -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs min-w-0">
      <!-- x / Δx view toggle -->
      <div class="flex items-center rounded border border-hairline overflow-hidden text-[10px] font-mono font-bold">
        <button
          on:click={() => (viewMode = 'stream')}
          class={`px-2 py-0.5 transition-colors ${viewMode === 'stream' ? 'bg-ink text-white' : 'bg-white text-ink-secondary hover:text-ink'}`}
          title="The running vector as it currently stands"
        >
          x
        </button>
        <button
          on:click={() => (viewMode = 'delta')}
          disabled={!deltaData}
          class={`px-2 py-0.5 transition-colors disabled:opacity-40 ${viewMode === 'delta' ? 'bg-ink text-white' : 'bg-white text-ink-secondary hover:text-ink'}`}
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
    <summary class="cursor-pointer text-[10px] font-sans font-semibold text-ink-muted hover:text-ink select-none">What's this?</summary>
    <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1">
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

        {#if hIdx % 2 === 1}
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
          Nothing here yet. Press Step or Play and each layer writes into these 288 channels
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
          class={`w-[16.6%] text-center transition-colors ${highlightedHead === hIdx ? 'text-ink font-black' : ''}`}
          title={`Head ${hIdx}'s slice of the vector: channels ${hIdx * 48} to ${hIdx * 48 + 47}`}
        >
          H{hIdx}
        </span>
      {/each}
    </div>
  </div>

  <!-- Hypothesis Evolution Tracker · Logit Lens -->
  <div class="mt-2.5 pt-2 border-t border-hairline text-[11px] font-mono min-w-0">
    <div class="flex flex-wrap items-center justify-between gap-y-1 gap-x-2 min-w-0">
      <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 min-w-0">
        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-800">Best Guess So Far</span>
        <span class="text-slate-300 font-bold">·</span>
        <span class="text-slate-600 font-sans font-medium">the model's top pick after each step</span>
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
      <div class="mt-1.5 flex items-center gap-1 overflow-x-auto scrollbar-none pb-0.5">
        {#each lensHistory as entry, i}
          {@const isLatest = i === lensHistory.length - 1}
          <div class="flex items-center gap-1 shrink-0">
            <div class="flex flex-col items-center px-1.5 py-0.5 rounded border transition-all duration-200 {isLatest ? 'border-ink bg-ink shadow-sm' : 'border-hairline bg-canvas-elevated'}">
              <span class="text-[8px] font-bold uppercase tracking-wider {isLatest ? 'text-zinc-400' : 'text-ink-muted'}">{entry.label}</span>
              <span class="text-[11px] font-bold font-sans whitespace-nowrap {isLatest ? 'text-white' : 'text-ink'}">{entry.top[0]?.tokenPiece ?? '·'}</span>
            </div>
            {#if !isLatest}
              <span class="text-slate-300 font-bold text-[10px]">→</span>
            {/if}
          </div>
        {/each}
      </div>

      <!-- Current step top-3 candidates with step-over-step logit deltas -->
      {#if latestLens}
        <div class="mt-1.5 flex items-center gap-1.5 overflow-x-auto scrollbar-none">
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
    {:else}
      <p class="mt-2 text-slate-800 font-medium leading-relaxed">Step the model forward. After each stage, this vector is scored against the vocabulary and the model's top guess appears here, evolving with depth</p>
    {/if}
  </div>
</div>
