<script lang="ts">
  import type { StepState } from '../../engine/forward';

  export let stepState: StepState | null = null;
  export let positionLabels: { text: string; id: number }[] = [];
  export let highlightedHead: number | null = null;
  export let severedHead: number | null = null;
  export let focusMode: 'attention' | 'memory' | 'prediction' | null = null;

  interface AttnSnapshot {
    layer: number;
    pos: number;
    nHeads: number;
    rows: number[][];
  }

  // Persist the most recent attention field so it stays visible during FFN steps
  let lastAttn: AttnSnapshot | null = null;

  $: if (stepState?.phase === 'post-attention' && stepState.attentionScores) {
    const t = stepState.attentionScores;
    const nHeads = t.shape[0];
    const seqLen = t.shape[1];
    const cols = stepState.position + 1;
    const rows: number[][] = [];
    for (let h = 0; h < nHeads; h++) {
      const row: number[] = [];
      for (let t2 = 0; t2 < cols; t2++) {
        row.push(t.data[t.offset + h * seqLen + t2]);
      }
      rows.push(row);
    }
    lastAttn = { layer: stepState.layer, pos: stepState.position, nHeads, rows };
  }

  function labelFor(t: number): string {
    const p = positionLabels[t];
    if (!p) return `#${t}`;
    const trimmed = p.text.trim();
    return trimmed.length > 0 ? `"${trimmed}"` : `#${p.id}`;
  }

  // cividis sequential ramp: perceptually-uniform, colorblind-safe attention weights
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

  function argmax(row: number[]): number {
    let best = 0;
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return best;
  }
</script>

<div class={`bg-white border rounded p-3 shadow-subtle h-full flex flex-col transition-all duration-300 ${focusMode === 'attention' ? 'border-amber-400 ring-2 ring-amber-400/50' : 'border-hairline'} ${focusMode === 'memory' ? 'border-hairline opacity-60 grayscale-[0.3]' : 'border-hairline'} ${focusMode === 'prediction' ? 'border-hairline opacity-60 grayscale-[0.3]' : 'border-hairline'}`}>
  <div class="flex flex-wrap items-center justify-between gap-y-1 gap-x-2 pb-2 mb-2 border-b border-hairline text-xs font-mono min-w-0">
    <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 min-w-0">
      <span class="text-ink font-serif font-bold text-[15px] tracking-tight" title="Technical term: Attention Field. Shows where each of the 6 attention heads looks back across earlier words.">Where It Looks Back</span>
      <span class="text-slate-300 font-bold">/</span>
      <span class="text-ink-secondary font-sans font-semibold text-xs">
        <span class="font-semibold">Attention Field</span> ·
        {#if lastAttn}
          Layer {lastAttn.layer} · looks back over {lastAttn.pos + 1} token{lastAttn.pos === 0 ? '' : 's'}
        {:else}
          6 heads · waiting for the first attention pass
        {/if}
      </span>
    </div>
    <span
      class="flex items-center gap-1 shrink-0"
      title="Color scale: each row is one head; brighter cells get more of that head's attention"
    >
      <span class="text-[9px] font-bold text-ink-muted uppercase tracking-wider">weak</span>
      <svg viewBox="0 0 64 8" class="w-16 h-2 rounded-[2px] overflow-hidden" aria-hidden="true">
        {#each Array(16) as _, i}
          <rect x={i * 4} y="0" width="4.5" height="8" fill={cividis(i / 15)} />
        {/each}
      </svg>
      <span class="text-[9px] font-bold text-ink-muted uppercase tracking-wider">strong</span>
    </span>
  </div>

  <p class="text-xs font-sans font-medium text-slate-900 mb-1">
    Each row is one head, showing where it looked.
  </p>
  <details class="text-xs mb-2">
    <summary class="inline-flex items-center gap-1.5 cursor-pointer text-xs font-sans font-semibold text-slate-700 hover:text-ink hover:bg-slate-100 active:bg-slate-200 px-2 py-1 rounded transition select-none">
      <span class="w-3.5 h-3.5 rounded-full bg-slate-200 text-slate-700 text-[10px] flex items-center justify-center font-bold">?</span>
      <span>What's this?</span>
    </summary>
    <svg viewBox="0 0 320 66" class="w-full max-w-sm h-auto mt-1.5" aria-hidden="true">
      <!-- one head looking back -->
      <rect x="14" y="30" width="38" height="20" rx="4" fill="#334155" />
      <text x="33" y="44" text-anchor="middle" font-size="8.5" font-family="sans-serif" fill="white">The</text>
      <rect x="60" y="30" width="46" height="20" rx="4" fill="#334155" />
      <text x="83" y="44" text-anchor="middle" font-size="8.5" font-family="sans-serif" fill="white">little</text>
      <rect x="114" y="30" width="40" height="20" rx="4" fill="#d97706" />
      <text x="134" y="44" text-anchor="middle" font-size="8.5" font-family="sans-serif" fill="white">girl</text>
      <path d="M134 28 C 134 8, 33 8, 33 28" fill="none" stroke="#d97706" stroke-width="1.6" opacity="0.8" />
      <path d="M134 28 C 134 14, 83 14, 83 28" fill="none" stroke="#d97706" stroke-width="1.2" opacity="0.5" />
      <text x="170" y="36" font-size="7.5" font-family="monospace" fill="#64748b">"girl" looks back at "The" and "little"</text>
      <text x="170" y="47" font-size="7.5" font-family="monospace" fill="#64748b">each row below = one head doing this</text>
    </svg>
    <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1.5 pl-1">
      Each row is one attention head, the same six heads as the Circuit Map squares. It shows which earlier tokens that head looks at. The outlined cell is its favorite, and the arrow names its top focus.
    </p>
  </details>

  {#if !lastAttn}
    <div class="relative flex-1 bg-slate-50 border border-slate-200 rounded shadow-inner p-3 overflow-hidden">
      <!-- Ghost rows: preview the six-head attention field before the first pass -->
      <div class="space-y-1" aria-hidden="true">
        {#each Array(6) as _, h}
          <div class="flex items-center gap-2">
            <span class="text-[9px] font-mono font-bold w-6 shrink-0 text-slate-300">H{h}</span>
            <div class="flex-1 flex gap-[2px]">
              {#each Array(8) as _, t}
                {@const w = Math.exp(-Math.pow(t - (1.4 + h * 0.95), 2) / 2.6)}
                <div class="h-4 flex-1 rounded-[1px]" style={`background: rgba(100,116,139,${0.08 + w * 0.4})`}></div>
              {/each}
            </div>
            <span class="w-24 shrink-0"></span>
          </div>
        {/each}
      </div>
      <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
        <span class="text-[11px] font-sans font-medium text-slate-900 bg-white/85 px-2 py-0.5 rounded text-center max-w-xs">
          Ghost rows preview the layout. Step into a layer's attention phase and each row lights up, showing where that head looked.
        </span>
      </div>
    </div>
  {:else}
    <div class="space-y-1">
      {#each lastAttn.rows as row, h}
        {@const isHighlighted = highlightedHead === h}
        {@const isSevered = severedHead === h}
        {@const focus = argmax(row)}
        <div class={`flex items-center gap-2 transition-opacity ${highlightedHead !== null && !isHighlighted ? 'opacity-30' : ''}`}>
          <span class={`text-[9px] font-mono font-bold w-6 shrink-0 ${isSevered ? 'text-accent-rust' : isHighlighted ? 'text-accent-bronze' : 'text-ink-muted'}`} title={isSevered ? `Head ${h} is turned off in the experiment` : undefined}>
            H{h}{isSevered ? ' ✕' : ''}
          </span>
          <div class="flex-1 flex gap-[2px]">
            {#each row as score, t}
              <div
                class="h-4 flex-1 rounded-[1px] transition-colors duration-300"
                style={`background: ${cividis(score)}; transition-delay: ${h * 35}ms${t === focus ? '; box-shadow: inset 0 0 0 1.5px rgba(17,18,21,0.85)' : ''}`}
                title={`H${h} → pos ${t} ${labelFor(t)}: ${score.toFixed(3)}`}
              ></div>
            {/each}
          </div>
          <!-- Dominant-focus annotation: where this head is actually looking -->
          <span class="w-24 shrink-0 text-right text-[9px] font-mono font-bold text-ink-secondary truncate">
            → <span class="text-ink">{labelFor(focus)}</span> <span style="color: #00224e">{(row[focus] * 100).toFixed(0)}%</span>
          </span>
        </div>
      {/each}

      <!-- Position labels -->
      {#if lastAttn.pos + 1 <= 24 && positionLabels.length > 0}
        <div class="flex items-center gap-2 pt-1">
          <span class="w-6 shrink-0"></span>
          <div class="flex-1 flex gap-[2px]">
            {#each Array(lastAttn.pos + 1) as _, t}
              <span class="flex-1 text-center text-[8px] font-mono font-bold text-ink-muted truncate">
                {labelFor(t)}
              </span>
            {/each}
          </div>
          <span class="w-24 shrink-0"></span>
        </div>
      {/if}
    </div>
  {/if}
</div>
