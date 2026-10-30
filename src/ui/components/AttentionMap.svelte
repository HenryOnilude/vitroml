<script lang="ts">
  import type { StepState } from '../../engine/forward';

  export let stepState: StepState | null = null;
  export let positionLabels: { text: string; id: number }[] = [];
  export let highlightedHead: number | null = null;

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

<div class="bg-white border border-hairline rounded p-3 shadow-subtle h-full flex flex-col">
  <div class="flex flex-wrap items-center justify-between gap-y-1 gap-x-2 pb-2 mb-2 border-b border-hairline text-xs font-mono min-w-0">
    <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 min-w-0">
      <span class="text-ink font-bold text-xs uppercase tracking-wider">Attention Field</span>
      <span class="text-slate-300 font-bold">/</span>
      <span class="text-ink-secondary font-sans font-semibold text-xs">
        {#if lastAttn}
          L{lastAttn.layer} · the token at position {lastAttn.pos} looks back over {lastAttn.pos + 1} token{lastAttn.pos === 0 ? '' : 's'}
        {:else}
          6 heads · waiting for the first attention pass
        {/if}
      </span>
    </div>
    <span class="text-[10px] font-bold text-ink-muted uppercase tracking-wider" title="Each row is one head; darker cells get more of that head's attention">
      darker = stronger focus
    </span>
  </div>

  <p class="text-xs font-sans font-medium text-slate-900 mb-1">
    Each row is one head, showing where it looked.
  </p>
  <details class="text-xs mb-2">
    <summary class="cursor-pointer text-[10px] font-sans font-semibold text-ink-muted hover:text-ink select-none">What's this?</summary>
    <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1">
      Each row is one attention head, the same six heads as the Circuit Map squares on the right. It shows which earlier tokens that head looks at. The outlined cell is its favorite, and the arrow names its top focus.
    </p>
  </details>

  {#if !lastAttn}
    <div class="flex-1 flex items-center justify-center bg-slate-50 border border-slate-200 rounded shadow-inner">
      <span class="text-[11px] font-sans font-medium text-slate-900 bg-white/80 px-2 py-0.5 rounded text-center max-w-xs">
        Nothing here yet. Step into a layer's attention phase and each row lights up, showing where that head looked.
      </span>
    </div>
  {:else}
    <div class="space-y-1">
      {#each lastAttn.rows as row, h}
        {@const isHighlighted = highlightedHead === h}
        {@const focus = argmax(row)}
        <div class={`flex items-center gap-2 transition-opacity ${highlightedHead !== null && !isHighlighted ? 'opacity-30' : ''}`}>
          <span class={`text-[9px] font-mono font-bold w-6 shrink-0 ${isHighlighted ? 'text-accent-bronze' : 'text-ink-muted'}`}>
            H{h}
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
