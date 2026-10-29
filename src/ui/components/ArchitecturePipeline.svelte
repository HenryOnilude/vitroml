<script lang="ts">
  import type { ExecutionState } from '../../bridge/controller';

  export let state: ExecutionState | null = null;
  export let currentToken: number = 1;
  export let inputPiece: string = '';

  const LAYERS = [0, 1, 2, 3, 4, 5];

  $: currentLayer = state?.currentStep?.layer ?? -1;
  $: currentPhase = state?.currentStep?.phase ?? null;
  $: isCompleted = state?.status === 'completed';
  $: isIdle = state?.status === 'idle' || !state;

  function isStepActive(layer: number, phase: 'post-attention' | 'post-ffn'): boolean {
    if (!state?.currentStep) return false;
    return currentLayer === layer && currentPhase === phase;
  }

  function isLayerPast(layer: number): boolean {
    if (isCompleted) return true;
    if (!state?.currentStep) return false;
    return layer < currentLayer;
  }

  // Token-centric flow: the data packet's fractional position along the bus.
  // Node indices: Input=0, Embed=1, L0..L5=2..7, Norm=8, Logits=9
  $: packetFrac = isCompleted
    ? 1
    : isIdle
    ? 0
    : (2 + currentLayer) / 9;
  $: isRunning = state?.status === 'running';
</script>

<div class="bg-white border border-hairline rounded p-3 shadow-subtle flex flex-col gap-2.5">
  <!-- Top Telemetry Row with High-Contrast Text -->
  <div class="flex items-center justify-between text-xs font-mono">
    <div class="flex items-center gap-2">
      <span class="text-ink font-bold text-xs uppercase tracking-wider">Execution Pipeline</span>
      <span class="text-slate-300 font-bold">•</span>
      <span class="text-ink-secondary font-sans font-medium text-xs">6 stops · look back, then refine, at each one</span>
    </div>

    <!-- Active State & Tensor Shape Badge -->
    <div class="flex items-center gap-2">
      {#if state?.currentStep}
        <span class="text-amber-950 font-bold bg-amber-100/80 border border-amber-300 px-2.5 py-0.5 rounded text-[11px] shadow-sm flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-amber-600 animate-pulse"></span>
          <span>Layer {state.currentStep.layer}</span>
          <span class="text-amber-400">/</span>
          <span class="uppercase tracking-wider">{state.currentStep.phase === 'post-attention' ? 'Attention · 6 heads' : 'FFN · memory lookup'}</span>
        </span>
      {:else if isCompleted}
        <span class="text-amber-950 font-bold bg-amber-100/80 border border-amber-300 px-2.5 py-0.5 rounded text-[11px] flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-amber-600"></span>
          <span>Done · next-word scores ready</span>
        </span>
      {:else}
        <span class="text-ink-secondary font-semibold text-[11px] font-mono bg-canvas-elevated px-2 py-0.5 rounded border border-hairline">
          Ready · waiting at the start
        </span>
      {/if}
    </div>
  </div>

  <p class="text-xs font-sans font-medium text-slate-900 leading-relaxed">
    This strip maps the journey one word takes to guess the next word. The amber dot is that word travelling through the model. <strong>Embed</strong> gives it a starting meaning. Each layer <strong>L0 to L5</strong> looks back at earlier words (<strong>Attn</strong>), then refines what it knows (<strong>FFN</strong>). <strong>Rescale</strong> tidies the numbers, and <strong>Scores</strong> turns it into a guess. Press Step or Play and watch the dot travel.
  </p>

  <!-- Sleek Continuous Architectural Bus -->
  <div class="relative flex items-center justify-between bg-canvas-elevated border border-hairline rounded p-2 overflow-x-auto scrollbar-none gap-2">
    <!-- Continuous horizontal guide line through all nodes -->
    <div class="absolute left-6 right-6 top-1/2 -translate-y-1/2 h-[1px] bg-slate-300 z-0"></div>

    <!-- Traveling data packet: the token embedding moving through the network -->
    <div
      class="absolute top-1/2 z-20 -translate-y-1/2 -translate-x-1/2 pointer-events-none transition-[left] duration-500 ease-out"
      style={`left: calc(1.5rem + (100% - 3rem) * ${packetFrac})`}
    >
      <span class={`relative block w-2.5 h-2.5 rounded-full transition-colors duration-300 ${
        isCompleted ? 'bg-amber-600' : isIdle ? 'bg-amber-500 animate-packet-idle' : 'bg-amber-500'
      } shadow-[0_0_8px_2px_rgba(217,119,6,0.45)]`}>
        {#if isRunning}
          <span class="absolute inset-0 rounded-full bg-amber-400 animate-ping opacity-70"></span>
        {/if}
      </span>
    </div>

    <!-- 1. Input Token Node -->
    <div
      class="relative z-10 flex flex-col items-center gap-1 bg-white border border-slate-300 px-3 py-1 rounded shadow-subtle shrink-0"
      title={`Token ID ${currentToken}`}
    >
      <span class="text-[10px] font-mono font-bold uppercase tracking-wider text-ink-tertiary">Input</span>
      <span class="text-xs font-mono font-bold text-ink max-w-[72px] truncate">
        {inputPiece ? `"${inputPiece}"` : `Tok ${currentToken}`}
      </span>
    </div>

    <!-- 2. Embedding Node -->
    <div
      title="Embedding: looks up the token's starting vector (288 numbers)"
      class={`relative z-10 flex flex-col items-center gap-1 px-3 py-1 rounded border text-xs font-mono transition-all duration-150 shrink-0 ${
      !isIdle
        ? 'bg-ink text-white font-bold border-ink shadow-sm'
        : 'bg-white border-slate-300 text-ink shadow-subtle font-semibold'
    }`}>
      <span class={`text-[10px] font-bold uppercase tracking-wider ${!isIdle ? 'text-zinc-300' : 'text-ink-tertiary'}`}>Embed</span>
      <span class="text-xs">288 numbers</span>
    </div>

    <!-- 3. The 6 Transformer Layers (Discrete High-Tech Nodes) -->
    {#each LAYERS as layer}
      {@const isLayerActive = currentLayer === layer && !isCompleted}
      {@const isPast = isLayerPast(layer)}
      {@const attnActive = isStepActive(layer, 'post-attention')}
      {@const ffnActive = isStepActive(layer, 'post-ffn')}

      <div
        title={`Layer ${layer}: attention (which tokens to look at) then FFN (memory lookup)`}
        class={`relative z-10 flex items-center gap-1 p-1 rounded border transition-all duration-150 shrink-0 ${
        isLayerActive
          ? 'bg-amber-100/80 border-amber-500 ring-2 ring-amber-200 shadow-sm'
          : isPast
          ? 'bg-white border-slate-400 text-ink shadow-subtle'
          : 'bg-white border-slate-300 text-ink-secondary'
      }`}>
        <span class="text-[11px] font-mono font-bold px-1 text-ink">L{layer}</span>

        <!-- Attention sub-phase pill -->
        <span
          title="Attention: heads decide which earlier tokens to look at"
          class={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-all duration-150 ${
          attnActive
            ? 'bg-amber-600 text-white shadow-sm'
            : isPast
            ? 'bg-slate-200 text-slate-900 font-bold'
            : 'bg-canvas-elevated text-ink-secondary'
        }`}>
          Attn
        </span>

        <!-- FFN sub-phase pill -->
        <span
          title="FFN (feed-forward network): the model's memory lookup. It mixes in learned patterns"
          class={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-all duration-150 ${
          ffnActive
            ? 'bg-amber-600 text-white shadow-sm'
            : isPast
            ? 'bg-slate-200 text-slate-900 font-bold'
            : 'bg-canvas-elevated text-ink-secondary'
        }`}>
          FFN
        </span>
      </div>
    {/each}

    <!-- 4. Final RMSNorm Node -->
    <div
      title="Final normalization: rescales the vector before scoring"
      class={`relative z-10 flex flex-col items-center gap-1 px-3 py-1 rounded border text-xs font-mono transition-all duration-150 shrink-0 ${
      isCompleted
        ? 'bg-ink text-white font-bold border-ink shadow-sm'
        : 'bg-white border-slate-300 text-ink shadow-subtle font-semibold'
    }`}>
      <span class={`text-[10px] font-bold uppercase tracking-wider ${isCompleted ? 'text-zinc-300' : 'text-ink-tertiary'}`}>Rescale</span>
      <span class="text-xs">evens out</span>
    </div>

    <!-- 5. Logits Output Projection Node -->
    <div
      title="Scores: every vocabulary token gets a score. The highest is the model's guess for the next word"
      class={`relative z-10 flex flex-col items-center gap-1 px-3 py-1 rounded border text-xs font-mono transition-all duration-150 shrink-0 ${
      isCompleted
        ? 'bg-amber-600 text-white font-bold border-amber-700 shadow-sm'
        : 'bg-white border-slate-300 text-ink shadow-subtle font-semibold'
    }`}>
      <span class={`text-[10px] font-bold uppercase tracking-wider ${isCompleted ? 'text-amber-100' : 'text-ink-tertiary'}`}>Scores</span>
      <span class="text-xs">32k words</span>
    </div>
  </div>
</div>
