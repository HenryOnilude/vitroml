<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let selectedLayer: number = 5;
  export let selectedHead: number = 4;
  export let ablatedLayer: number | null = 5;
  export let ablatedHead: number | null = 4;
  export let currentActiveLayer: number = -1;
  export let hoveredHead: number | null = null;
  // Measured importance: importance[l][h] = ‖Δlogits‖₂ when head is zero-ablated.
  // null = not yet computed for that head.
  export let importance: (number | null)[][] | null = null;

  $: importanceMax = (() => {
    let m = 0;
    if (importance) {
      for (const row of importance) for (const v of row) if (v !== null && v > m) m = v;
    }
    return m;
  })();

  const dispatch = createEventDispatcher<{
    select: { layer: number; head: number };
    hover: { layer: number; head: number | null };
  }>();

  const LAYERS = [0, 1, 2, 3, 4, 5];
  const HEADS = [0, 1, 2, 3, 4, 5];

  function handleHeadClick(l: number, h: number) {
    selectedLayer = l;
    selectedHead = h;
    dispatch('select', { layer: l, head: h });
  }

  function handleMouseEnter(l: number, h: number) {
    hoveredHead = h;
    dispatch('hover', { layer: l, head: h });
  }

  function handleMouseLeave() {
    hoveredHead = null;
    dispatch('hover', { layer: -1, head: null });
  }
</script>

<div class="bg-white border border-hairline rounded p-3 shadow-subtle">
  <div class="flex flex-wrap items-center justify-between gap-y-1.5 gap-x-2 pb-2 mb-2 border-b border-hairline text-xs font-mono min-w-0">
    <div class="flex flex-wrap items-center gap-2 min-w-0">
      <span class="text-ink font-bold text-xs uppercase tracking-wider">Circuit Map</span>
      <span class="text-hairline-bright">•</span>
      <span class="text-ink-secondary font-sans text-xs">36 Attention Heads</span>
    </div>

    <!-- Minimal legend -->
    <div class="text-[11px] text-ink-tertiary flex flex-wrap items-center gap-x-3 gap-y-1 font-mono min-w-0">
      <span class="flex items-center gap-1.5">
        <span class="w-2.5 h-2.5 rounded bg-ink shadow-sm"></span>
        <span class="text-ink-secondary">Selected</span>
      </span>
      <span class="flex items-center gap-1.5">
        <span class="w-2.5 h-2.5 rounded shadow-sm" style="background: #b91c1c; background-image: repeating-linear-gradient(45deg, rgba(255,255,255,0.25) 0 2px, transparent 2px 4px)"></span>
        <span class="text-accent-rust font-medium">Off</span>
      </span>
      <span class="flex items-center gap-1.5">
        <span class="w-2.5 h-2.5 rounded border border-hairline-bright" style="background: rgba(17,18,21,0.35)"></span>
        <span class="text-ink-secondary">Impact if off</span>
      </span>
    </div>
  </div>

  <p class="text-xs font-sans font-medium text-slate-900 mb-1">
    Click a square to switch that head off.
  </p>
  <details class="text-xs mb-2">
    <summary class="cursor-pointer text-[10px] font-sans font-semibold text-ink-muted hover:text-ink select-none">What's this?</summary>
    <p class="font-sans font-medium text-slate-900 leading-relaxed mt-1">
      Each square is one attention head (<span class="font-mono">layer:head</span>). The shading shows how much the output changes when that head is off. Darker means it matters more. Click a square to switch that head off and watch what changes below.
    </p>
  </details>

  <!-- 6x6 Head Matrix Grid with Hover Brushing -->
  <div class="grid grid-cols-7 gap-1 text-xs font-mono items-center">
    <!-- Header Row -->
    <div class="text-[10px] text-ink-muted font-medium text-center"></div>
    {#each HEADS as h}
      <div class={`text-[11px] font-bold text-center transition-colors ${
        hoveredHead === h ? 'text-ink' : 'text-ink-tertiary'
      }`}>
        H{h}
      </div>
    {/each}

    <!-- 6 Layer Rows -->
    {#each LAYERS as l}
      <div class={`text-[11px] font-bold text-center transition-colors ${
        currentActiveLayer === l ? 'text-ink font-extrabold' : 'text-ink-muted'
      }`}>
        L{l}
      </div>

      {#each HEADS as h}
        {@const isSelected = selectedLayer === l && selectedHead === h}
        {@const isAblated = ablatedLayer === l && ablatedHead === h}
        {@const isLayerActive = currentActiveLayer === l}
        {@const imp = importance?.[l]?.[h] ?? null}
        {@const impAlpha = imp !== null && importanceMax > 0 ? 0.05 + 0.4 * (imp / importanceMax) : 0}

        <button
          type="button"
          on:click={() => handleHeadClick(l, h)}
          on:mouseenter={() => handleMouseEnter(l, h)}
          on:mouseleave={handleMouseLeave}
          class={`h-8 rounded text-[11px] font-mono font-medium transition-all duration-100 flex flex-col items-center justify-center border active:scale-95 ${
            isAblated
              ? 'border-accent-rust text-white shadow-sm'
              : isSelected
              ? 'bg-ink border-ink text-white font-bold shadow-sm'
              : isLayerActive
              ? 'border-ink text-ink font-bold'
              : 'hover:bg-white border-hairline hover:border-hairline-bright text-ink-secondary hover:text-ink'
          }`}
          style={isAblated
            ? 'background: #b91c1c; background-image: repeating-linear-gradient(45deg, rgba(255,255,255,0.22) 0 2px, transparent 2px 4px)'
            : isSelected
            ? ''
            : `background: rgba(17,18,21,${impAlpha})`}
          title={`Layer ${l}, Head ${h} · Channels [${h * 48}..${(h + 1) * 48 - 1}]${imp !== null ? ` · output shifts ${imp.toFixed(1)} when off` : ''} · click to switch off`}
        >
          <span>{l}:{h}</span>
          <span class={`text-[8px] leading-none ${isAblated || isSelected ? 'opacity-85' : 'text-ink-muted'}`}>
            {imp !== null ? `Δ${imp >= 100 ? imp.toFixed(0) : imp.toFixed(1)}` : `${h * 48}d`}
          </span>
        </button>
      {/each}
    {/each}
  </div>

  <!-- Δ definition: the number only means something once you know what was measured -->
  <div class="mt-2 pt-2 border-t border-hairline text-[10px] font-mono text-slate-800 font-medium leading-snug">
    Δ = how much the output scores change when that head is switched off. Darker = more important
  </div>
</div>
