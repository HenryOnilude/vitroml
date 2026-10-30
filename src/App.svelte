<script lang="ts">
  import { onMount } from 'svelte';
  import { parseCheckpoint } from './engine/loader';
  import { parseTokenizer, type Tokenizer } from './engine/tokenizer';
  import type { ModelConfig, ModelWeights } from './engine/model';
  import { HookRegistry } from './engine/hooks';
  import { ForwardController } from './bridge/controller';
  import { argmaxToken, probeBaselineLogits, measureHeadAblationL2 } from './engine/analysis';

  import WorkbenchToolbar from './ui/components/WorkbenchToolbar.svelte';
  import HeroSection from './ui/components/HeroSection.svelte';
  import NarrativeBar from './ui/components/NarrativeBar.svelte';
  import ArchitecturePipeline from './ui/components/ArchitecturePipeline.svelte';
  import ResidualStreamView from './ui/components/ResidualStreamView.svelte';
  import InterventionPanel from './ui/components/InterventionPanel.svelte';
  import AttentionMap from './ui/components/AttentionMap.svelte';

  let isLoading: boolean = true;
  let loadingProgress: string = 'Initializing engine...';
  let loadError: string | null = null;

  let config: ModelConfig | null = null;
  let weights: ModelWeights | null = null;
  let tokenizer: Tokenizer | null = null;
  let controller: ForwardController | null = null;

  let currentTokens: number[] = [];
  let currentToken: number = 1;   // input token at the active position (pos 0 = BOS)
  let currentPos: number = 0;

  // Autoregressive state: input ids by position = [BOS, ...prompt, ...generated]
  let inputSequence: number[] = [1];
  let generatedTokens: { id: number; piece: string }[] = [];
  let autoGen: boolean = false;
  let lastSampledPiece: string | null = null;
  const MAX_GENERATED = 16;
  const EOS_ID = 2;

  // Cross-view brushing state: hovered head index (0..5)
  let brushedHead: number | null = null;

  let stateStore: ForwardController['state'] | null = null;
  $: stateStore = controller ? controller.state : null;
  $: currentActiveLayer = $stateStore?.currentStep?.layer ?? -1;

  // Keep App's token/pos mirror in sync with whatever the controller is processing
  $: if ($stateStore?.currentStep) {
    currentPos = $stateStore.currentStep.position;
    currentToken = inputSequence[currentPos] ?? currentToken;
  }

  // Token labels per position, for attention-map columns and the input node
  $: positionLabels = [
    { text: '<s>', id: 1 },
    ...currentTokens.map((id, i) => ({
      text: tokenizer ? tokenizer.decode(i === 0 ? 1 : currentTokens[i - 1], id) : '',
      id,
    })),
    ...generatedTokens.map((g) => ({ text: g.piece, id: g.id })),
  ];

  $: currentInputLabel = (() => {
    const p = positionLabels[currentPos];
    if (!p) return `#${currentToken}`;
    const t = p.text.trim();
    return t.length > 0 ? t : `#${p.id}`;
  })();

  // Head-importance saliency map: ‖Δlogits‖₂ per head, measured on load
  let headImportance: (number | null)[][] | null = null;

  async function runImportanceScan() {
    if (!config || !weights) return;
    const baseline = probeBaselineLogits(1, 0, config, weights);
    const grid: (number | null)[][] = Array.from({ length: config.nLayers }, () =>
      Array(config.nHeads).fill(null)
    );
    headImportance = grid;
    for (let l = 0; l < config.nLayers; l++) {
      for (let h = 0; h < config.nHeads; h++) {
        grid[l][h] = measureHeadAblationL2(baseline, 1, 0, l, h, config, weights);
      }
      headImportance = grid; // reveal layer-by-layer
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // Autoregressive continuation: on completion, prefill prompt tokens first,
  // then feed back argmax predictions. autoGen keeps the loop running.
  $: if (autoGen && $stateStore?.status === 'completed' && $stateStore.result) {
    advanceGeneration('auto');
  }

  function advanceGeneration(resume: 'step' | 'play' | 'auto') {
    const result = $stateStore?.result;
    if (!result || !controller || !tokenizer) return;

    const nextPos = inputSequence.length;
    const prefillEnd = 1 + currentTokens.length; // BOS + all prompt tokens
    const isPrefill = nextPos < prefillEnd;
    const nextId = isPrefill ? currentTokens[nextPos - 1] : argmaxToken(result.logits);

    if (isPrefill) {
      lastSampledPiece = null;
    } else {
      const prevId = inputSequence[inputSequence.length - 1];
      const piece = tokenizer.decode(prevId, nextId);
      generatedTokens = [...generatedTokens, { id: nextId, piece }];
      lastSampledPiece = piece;
    }

    inputSequence = [...inputSequence, nextId];
    const ok = controller.advance(nextId);
    if (!ok) {
      autoGen = false;
      return;
    }
    currentToken = nextId;
    currentPos = controller.getPosition();

    if (resume === 'play' || resume === 'auto') {
      const keepGoing = autoGen && generatedTokens.length < MAX_GENERATED && nextId !== EOS_ID;
      autoGen = keepGoing;
      if (keepGoing) controller.play();
    } else if (resume === 'step') {
      controller.step();
    }
  }

  function handleAdvance(event: CustomEvent<{ resume: 'step' | 'play' }>) {
    if (event.detail.resume === 'play') autoGen = true;
    advanceGeneration(event.detail.resume);
  }

  function resetGenerationState() {
    autoGen = false;
    generatedTokens = [];
    inputSequence = [1, ...currentTokens];
    lastSampledPiece = null;
    currentToken = 1;
    currentPos = 0;
  }

  onMount(async () => {
    try {
      loadingProgress = 'Loading tokenizer...';
      const baseUrl = import.meta.env.BASE_URL || '/';
      const tokRes = await fetch(`${baseUrl}models/tokenizer.bin`);
      if (!tokRes.ok) {
        throw new Error(`Failed to load tokenizer: ${tokRes.status} ${tokRes.statusText}`);
      }
      const tokBuf = await tokRes.arrayBuffer();
      tokenizer = parseTokenizer(tokBuf, 32000);

      loadingProgress = 'Loading stories15M.bin (58 MB)...';
      const modelRes = await fetch(`${baseUrl}models/stories15M.bin`);
      if (!modelRes.ok) {
        throw new Error(`Failed to load model weights: ${modelRes.status} ${modelRes.statusText}`);
      }
      const modelBuf = await modelRes.arrayBuffer();

      loadingProgress = 'Mapping zero-copy tensors...';
      const parsed = parseCheckpoint(modelBuf);
      config = parsed.config;
      weights = parsed.weights;

      controller = new ForwardController(config, weights, new HookRegistry());
      controller.start(currentToken, currentPos);

      isLoading = false;
      runImportanceScan();
    } catch (err: unknown) {
      console.error('Model initialization error:', err);
      loadError = err instanceof Error ? err.message : String(err);
      isLoading = false;
    }
  });

  function handleTokensChange(event: CustomEvent<{ prompt: string; tokens: number[] }>) {
    currentTokens = event.detail.tokens;
    resetGenerationState();
    if (controller && !isLoading) {
      controller.start(1, 0); // BOS at position 0, llama2.c generate() convention
    }
  }

  function handleHeadHover(event: CustomEvent<{ layer: number; head: number | null }>) {
    brushedHead = event.detail.head;
  }
</script>

<div class="min-h-screen bg-canvas text-ink flex flex-col font-sans">
  <!-- Minimalist High-End Masthead -->
  <header class="border-b border-hairline bg-white/80 backdrop-blur sticky top-0 z-50 shadow-subtle">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 min-h-[3rem] py-1.5 flex flex-wrap items-center justify-between gap-2 min-w-0">
      <!-- Left: Brand / Wordmark -->
      <div class="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 min-w-0">
        <span class="font-extrabold text-sm tracking-tight text-ink font-mono uppercase">
          Vitro
        </span>
        <span class="text-hairline-bright font-mono text-xs">/</span>
        <span class="text-xs text-ink-secondary font-semibold truncate">
          stories15M · Mechanistic Interpretability Workbench
        </span>
      </div>

      <!-- Right: Technical Status -->
      <div class="flex items-center gap-3 text-xs font-mono">
        {#if config}
          <div class="flex items-center gap-1.5 text-[11px] bg-slate-100 px-2.5 py-0.5 rounded border border-slate-300">
            <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
            <span class="text-ink-secondary font-bold">Engine Live</span>
          </div>
        {/if}
      </div>
    </div>
  </header>

  <!-- Main Viewport -->
  <main class="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-4">
    {#if isLoading}
      <!-- Loading State -->
      <div class="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div class="w-6 h-6 border-2 border-hairline-bright border-t-ink rounded-full animate-spin"></div>
        <div class="text-xs font-mono text-ink-secondary font-bold tracking-widest uppercase">{loadingProgress}</div>
      </div>
    {:else if loadError}
      <!-- Error State -->
      <div class="p-8 bg-white border border-rose-200 rounded max-w-md mx-auto text-center mt-12 shadow-sm">
        <div class="text-rose-700 text-xs font-bold uppercase tracking-wider mb-2 font-mono">Initialization Error</div>
        <div class="text-xs font-mono text-ink-secondary bg-canvas-elevated p-3 rounded border border-hairline mb-4 break-words text-left">
          {loadError}
        </div>
        <button
          on:click={() => window.location.reload()}
          class="px-4 py-1.5 bg-ink hover:bg-black text-white font-bold rounded text-xs transition"
        >
          Reload Session
        </button>
      </div>
    {:else}
      <!-- Unified Architectural Layout (Zero Fluff) -->
      <div class="space-y-3">
        <!-- 0. Hero: what this is and what you can do with it -->
        <HeroSection />

        <!-- 1. Interactive Control Strip -->
        <WorkbenchToolbar
          {tokenizer}
          {controller}
          promptText="Once upon a time"
          {currentPos}
          {generatedTokens}
          on:tokensChange={handleTokensChange}
          on:advance={handleAdvance}
          on:userPlay={() => (autoGen = true)}
          on:userPause={() => (autoGen = false)}
          on:userStep={() => (autoGen = false)}
          on:userReset={resetGenerationState}
        />

        <!-- 2. Live Narrative Strip: what the model is doing right now -->
        <NarrativeBar
          state={$stateStore ?? null}
          {currentToken}
          {currentPos}
          sampledPiece={lastSampledPiece}
        />

        <!-- 3. Hardware Model Backbone Track -->
        <ArchitecturePipeline
          state={$stateStore ?? null}
          {currentToken}
          inputPiece={currentInputLabel}
        />

        <!-- 4. Dual Data Viewports: watch the computation (left), then intervene (right) -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3 items-stretch">
          <div class="flex flex-col gap-3 min-w-0">
            <div class="flex flex-wrap items-center gap-2 px-1 min-w-0">
              <span class="text-[9px] font-mono font-bold uppercase tracking-[0.18em] text-ink-muted">Act 1 · Inside the computation</span>
              <span class="flex-1 h-px bg-hairline min-w-[1rem]"></span>
              <span class="text-[10px] font-sans text-ink-muted">the same six heads appear in every view</span>
            </div>
            <ResidualStreamView
              stepState={$stateStore?.currentStep ?? null}
              highlightedHead={brushedHead}
              {config}
              {weights}
              {tokenizer}
              {currentToken}
            />

            <div class="flex-1 min-w-0">
              <AttentionMap
                stepState={$stateStore?.currentStep ?? null}
                {positionLabels}
                highlightedHead={brushedHead}
              />
            </div>
          </div>

          <div class="space-y-3 min-w-0">
            <div class="flex flex-wrap items-center gap-2 px-1 min-w-0">
              <span class="text-[9px] font-mono font-bold uppercase tracking-[0.18em] text-ink-muted">Act 2 · Run the experiment</span>
              <span class="flex-1 h-px bg-hairline min-w-[1rem]"></span>
              <span class="text-[10px] font-sans text-ink-muted">switch a head off, measure what breaks</span>
            </div>
            <InterventionPanel
              {config}
              {weights}
              {tokenizer}
              {currentActiveLayer}
              importance={headImportance}
              tokens={inputSequence}
              activePos={currentPos}
              on:headHover={handleHeadHover}
            />
          </div>
        </div>
      </div>
    {/if}
  </main>

  <!-- High-Contrast Technical Telemetry Footer -->
  <footer class="border-t border-hairline py-3 px-4 sm:px-6 bg-white text-xs font-mono">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2 min-w-0">
      <span class="text-slate-800 font-bold">15.2M Parameters · 6 Layers · 6 Heads · d=288</span>
      <span class="text-slate-900 font-extrabold">Token {currentToken} · Position {currentPos}</span>
    </div>
  </footer>
</div>
