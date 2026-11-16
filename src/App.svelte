<script lang="ts">
  import { onMount, tick } from 'svelte';
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
  import GuidedTour from './ui/components/GuidedTour.svelte';

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

  // Active attention scores preserved across steps for Attention Rays
  let lastAttnScores: import('./engine/tensor').Tensor | null = null;
  let lastAttnLayer: number = 0;
  $: if ($stateStore?.currentStep?.phase === 'post-attention' && $stateStore.currentStep.attentionScores) {
    lastAttnScores = $stateStore.currentStep.attentionScores;
    lastAttnLayer = $stateStore.currentStep.layer;
  }

  $: activeHeadRays = (() => {
    if (!lastAttnScores || currentPos <= 0) return null;
    const h = brushedHead ?? activeLensHead ?? activeSeveredHead ?? 0;
    const seqLen = lastAttnScores.shape[1];
    const weights: number[] = [];
    const off = lastAttnScores.offset + h * seqLen;
    for (let t = 0; t <= currentPos; t++) {
      weights.push(lastAttnScores.data[off + t]);
    }
    return { layer: lastAttnLayer, head: h, weights };
  })();

  function handleLensHeadFocus(event: CustomEvent<{ layer: number; head: number } | null>) {
    if (!event.detail) {
      activeLensHead = null;
      return;
    }
    activeLensHead = event.detail.head;
    brushedHead = event.detail.head;
  }

  // Phase Focus: guide user attention to the active computation stage
  $: focusMode = (() => {
    const phase = $stateStore?.currentStep?.phase;
    if (phase === 'post-attention') return 'attention' as const;
    if (phase === 'post-ffn') return 'memory' as const;
    if ($stateStore?.status === 'completed') return 'prediction' as const;
    return null;
  })();

  // Mobile act switcher: on <lg only one column is visible at a time
  let mobileAct: 1 | 2 = 1;

  // Act 2 reveal: collapsed teaser until clicked or the first playthrough ends
  let act2Open = false;
  let act2Hint = false;
  let seenFirstComplete = false;
  let showTryAnother = false;
  $: if ($stateStore?.status === 'completed' && !seenFirstComplete) {
    seenFirstComplete = true;
    if (!act2Open) {
      act2Hint = true;
      tick().then(() => {
        document.getElementById('panel-act2-teaser')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    }
  }
  function openAct2() {
    act2Open = true;
    act2Hint = false;
  }

  // Curated example: per-preset head whose ablation changes the guess (from the
  // sweep). "Once upon a time" has no flipping head, so its biggest confidence
  // drop is used instead and said so. Fallback is the sweep's strongest head.
  const CURATED_HEADS: Record<string, { layer: number; head: number; flipped: boolean }> = {
    'Once upon a time': { layer: 0, head: 1, flipped: false },
    'The little girl saw a bird': { layer: 5, head: 3, flipped: true },
  };
  let currentPrompt = 'Once upon a time';
  $: curated = CURATED_HEADS[currentPrompt] ?? { layer: 5, head: 4, flipped: false };
  $: curatedKnown = currentPrompt in CURATED_HEADS;
  $: step2Label = curated.flipped
    ? `Switch off L${curated.layer}:H${curated.head}, the head that changes the answer.`
    : curatedKnown
    ? `Switch off L${curated.layer}:H${curated.head}, the biggest lever on this guess.`
    : `Switch off L${curated.layer}:H${curated.head}, the biggest lever we found.`;

  let highlightedTarget: 'controls' | 'circuit' | 'result' | null = null;
  let highlightTimer: ReturnType<typeof setTimeout> | null = null;
  let interventionRef: InterventionPanel | null = null;

  let activeSeveredHead: number | null = null;
  let activeLensHead: number | null = null;

  function handleWhySelectHead(event: CustomEvent<{ layer: number; head: number }>) {
    const { layer, head } = event.detail;
    mobileAct = 2;
    activeSeveredHead = head;
    openAct2();
    interventionRef?.selectHead(layer, head);
  }

  let tourOpen = false;

  async function handleGuideStep(event: CustomEvent<1 | 2 | 3>) {
    const step = event.detail;
    if (step === 2 || step === 3) {
      mobileAct = 2;
      openAct2();
      await tick();
    }
    if (step === 2) {
      activeSeveredHead = curated.head;
      interventionRef?.selectHead(curated.layer, curated.head);
      showTryAnother = true;
    }
    const targetMap = {
      1: { key: 'controls' as const, id: 'panel-controls' },
      2: { key: 'circuit' as const, id: 'panel-circuit-map' },
      3: { key: 'result' as const, id: 'panel-experiment-result' },
    }[step];

    highlightedTarget = targetMap.key;
    if (highlightTimer) clearTimeout(highlightTimer);
    highlightTimer = setTimeout(() => {
      highlightedTarget = null;
    }, 2000);

    await tick();
    const el = document.getElementById(targetMap.id);
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

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

  const MODEL_CACHE = 'vitroml-models-v1';

  // Streams a binary with live download percentage, then persists it in the
  // Cache API so repeat visits skip the 58 MB download entirely.
  async function fetchBinary(url: string, expectedBytes: number, label: string, hint: string): Promise<ArrayBuffer> {
    if ('caches' in window) {
      try {
        const cache = await caches.open(MODEL_CACHE);
        const hit = await cache.match(url);
        if (hit) {
          loadingProgress = `${label} loaded from local cache`;
          return await hit.arrayBuffer();
        }
      } catch { /* Cache API unavailable; fall through to network */ }
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load ${label}: ${res.status} ${res.statusText}`);
    if (!res.body) return await res.arrayBuffer();

    const total = Number(res.headers.get('content-length')) || expectedBytes;
    const reader = res.body.getReader();
    const parts: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      received += value.byteLength;
      const pct = Math.min(100, Math.floor((received / total) * 100));
      loadingProgress = `Downloading ${label} ${pct}%${hint}`;
    }
    const buf = new Uint8Array(received);
    let offset = 0;
    for (const part of parts) {
      buf.set(part, offset);
      offset += part.byteLength;
    }

    if ('caches' in window) {
      try {
        const cache = await caches.open(MODEL_CACHE);
        await cache.put(url, new Response(buf.slice().buffer, { headers: { 'Content-Type': 'application/octet-stream' } }));
      } catch { /* storage full or blocked; page still works */ }
    }
    return buf.buffer;
  }

  onMount(async () => {
    try {
      const baseUrl = import.meta.env.BASE_URL || '/';
      loadingProgress = 'Loading tokenizer...';
      const tokBuf = await fetchBinary(`${baseUrl}models/tokenizer.bin`, 430_000, 'tokenizer', '');
      tokenizer = parseTokenizer(tokBuf, 32000);

      const modelBuf = await fetchBinary(`${baseUrl}models/stories15M.bin`, 61_000_000, 'stories15M.bin', ' · 58 MB, first visit only');

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

  // Chapter compass: the active chapter is the last marker above 40% of the viewport
  let activeChapter = 1;
  onMount(() => {
    const updateChapter = () => {
      const markers = document.querySelectorAll('[data-chapter]');
      let current = 1;
      for (const m of markers) {
        if (m.getBoundingClientRect().top <= window.innerHeight * 0.4) {
          current = Number((m as HTMLElement).dataset.chapter);
        }
      }
      activeChapter = current;
    };
    window.addEventListener('scroll', updateChapter, { passive: true });
    updateChapter();
    return () => window.removeEventListener('scroll', updateChapter);
  });

  function handleTokensChange(event: CustomEvent<{ prompt: string; tokens: number[] }>) {
    currentTokens = event.detail.tokens;
    currentPrompt = event.detail.prompt;
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
        <span class="font-serif font-black text-lg tracking-tight text-ink">
          Vitro
        </span>
        <span class="text-hairline-bright font-mono text-xs">/</span>
        <span class="text-xs text-ink-secondary font-semibold truncate" title="Mechanistic interpretability workbench for stories15M">
          stories15M · Inside an AI Storyteller
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
      <div class="space-y-3 relative">
        <!-- Story spine: a continuous rail the chapter numbers sit on -->
        <div class="absolute left-[17px] top-0 bottom-0 w-[2px] bg-slate-300/60 pointer-events-none" aria-hidden="true"></div>
        <!-- 0. Hero: what this is and what you can do with it -->
        <HeroSection {step2Label} on:guideStep={handleGuideStep} on:tour={() => (tourOpen = true)} />

        <!-- Chapter 1: give the model a story start -->
        <div id="chapter-1" class="flex flex-wrap items-center gap-2.5 px-1 min-w-0" data-chapter="1">
          <span class="w-7 h-7 rounded bg-ink text-white text-xs font-mono font-bold flex items-center justify-center shrink-0 relative z-10">1</span>
          <span class="font-serif font-bold text-lg text-ink tracking-tight">Give it a story start</span>
          <span class="flex-1 h-px bg-hairline min-w-[1rem]"></span>
          <span class="text-[10px] font-sans text-ink-muted">type a beginning, then press play</span>
        </div>
        <WorkbenchToolbar
          {tokenizer}
          {controller}
          promptText="Once upon a time"
          {currentPos}
          {generatedTokens}
          highlighted={highlightedTarget === 'controls'}
          attentionRays={activeHeadRays}
          {focusMode}
          on:tokensChange={handleTokensChange}
          on:advance={handleAdvance}
          on:userPlay={() => (autoGen = true)}
          on:userPause={() => (autoGen = false)}
          on:userStep={() => (autoGen = false)}
          on:userReset={resetGenerationState}
        />

        <!-- Chapter 2: watch one word travel through the model, narrated live -->
        <div id="chapter-2" class="flex flex-wrap items-center gap-2.5 px-1 min-w-0" data-chapter="2">
          <span class="w-7 h-7 rounded bg-ink text-white text-xs font-mono font-bold flex items-center justify-center shrink-0 relative z-10">2</span>
          <span class="font-serif font-bold text-lg text-ink tracking-tight">Watch it think</span>
          <span class="flex-1 h-px bg-hairline min-w-[1rem]"></span>
          <span class="text-[10px] font-sans text-ink-muted">one word's journey through 6 layers, live</span>
        </div>
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

        <!-- Mobile act switcher: pick one act at a time on narrow screens -->
        <div class="lg:hidden flex rounded border border-hairline overflow-hidden text-xs font-mono font-bold bg-white shadow-subtle min-h-[44px]">
          <button
            type="button"
            on:click={() => (mobileAct = 1)}
            class={`flex-1 min-h-[44px] px-3 py-2.5 transition-colors flex items-center justify-center active:scale-[0.99] ${mobileAct === 1 ? 'bg-ink text-white font-extrabold' : 'text-ink-secondary hover:text-ink hover:bg-slate-50'}`}
          >
            Act 1 · Inside
          </button>
          <button
            type="button"
            on:click={() => (mobileAct = 2)}
            class={`flex-1 min-h-[44px] px-3 py-2.5 transition-colors border-l border-hairline flex items-center justify-center active:scale-[0.99] ${mobileAct === 2 ? 'bg-ink text-white font-extrabold' : 'text-ink-secondary hover:text-ink hover:bg-slate-50'}`}
          >
            Act 2 · Experiment
          </button>
        </div>

        <!-- 4. Act 1: watch the computation with both instruments side by side -->
        <div class={`space-y-3 min-w-0 ${mobileAct === 1 ? '' : 'hidden'} lg:block`}>
          <div id="chapter-3" class="flex flex-wrap items-center gap-2.5 px-1 min-w-0" data-chapter="3">
            <span class="w-7 h-7 rounded bg-ink text-white text-xs font-mono font-bold flex items-center justify-center shrink-0 relative z-10">3</span>
            <span class="font-serif font-bold text-lg text-ink tracking-tight">Act 1 · Look inside</span>
            <span class="flex-1 h-px bg-hairline min-w-[1rem]"></span>
            <span class="text-[10px] font-sans text-ink-muted">the same six heads appear in every view</span>
          </div>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-3 items-stretch">
            <ResidualStreamView
              stepState={$stateStore?.currentStep ?? null}
              highlightedHead={brushedHead}
              severedHead={act2Open ? activeSeveredHead : null}
              {focusMode}
              {config}
              {weights}
              {tokenizer}
              {currentToken}
              on:selectHead={handleWhySelectHead}
              on:headFocus={handleLensHeadFocus}
            />

            <AttentionMap
              stepState={$stateStore?.currentStep ?? null}
              {positionLabels}
              highlightedHead={brushedHead}
              severedHead={act2Open ? activeSeveredHead : null}
              {focusMode}
            />
          </div>
        </div>

        <!-- 5. Act 2: the experiment, a full-width section below -->
        <div class={`space-y-3 min-w-0 ${mobileAct === 2 ? '' : 'hidden'} lg:block`}>
          <div id="chapter-4" class="flex flex-wrap items-center gap-2.5 px-1 min-w-0" data-chapter="4">
            <span class="w-7 h-7 rounded bg-ink text-white text-xs font-mono font-bold flex items-center justify-center shrink-0 relative z-10">4</span>
            <span class="font-serif font-bold text-lg text-ink tracking-tight">Act 2 · Break it</span>
            <span class="flex-1 h-px bg-hairline min-w-[1rem]"></span>
            <span class="text-[10px] font-sans text-ink-muted">switch a head off, measure what breaks</span>
          </div>
          {#if !act2Open}
            <button
              type="button"
              id="panel-act2-teaser"
              on:click={openAct2}
              class={`w-full flex items-center justify-between gap-4 text-left bg-white border rounded px-4 py-3.5 shadow-subtle transition-all duration-300 hover:border-amber-400 hover:shadow-md group ${
                act2Hint ? 'ring-2 ring-amber-400/70 border-amber-400 animate-pulse' : 'border-hairline'
              }`}
            >
              <span class="min-w-0">
                <span class="text-[10px] font-mono font-bold uppercase tracking-[0.18em] text-accent-bronze">Act 2</span>
                <span class="block text-sm font-bold text-ink font-serif mt-0.5">Now try breaking it</span>
                <span class="block text-[11px] text-ink-secondary font-sans mt-0.5">
                  {#if act2Hint}
                    Now switch a head off and see what changes.
                  {:else}
                    Switch one attention head off and see what changes.
                  {/if}
                </span>
              </span>
              <span class="shrink-0 inline-flex items-center gap-1.5 text-xs font-bold font-mono text-accent-bronze group-hover:text-amber-800 border border-amber-300 group-hover:border-amber-500 rounded px-3 py-2 transition-colors">
                Open →
              </span>
            </button>
          {:else}
            {#if showTryAnother}
              <div class="flex items-center justify-between gap-2 bg-amber-50 border border-amber-200 rounded px-2 py-1">
                <span class="text-[10px] font-sans font-medium text-amber-900">Now try another square.</span>
                <button
                  type="button"
                  on:click={() => (showTryAnother = false)}
                  class="text-amber-700 hover:text-amber-950 text-[10px] font-mono font-bold px-1"
                  aria-label="Dismiss hint"
                >✕</button>
              </div>
            {/if}
            <InterventionPanel
              bind:this={interventionRef}
              {config}
              {weights}
              {tokenizer}
              {currentActiveLayer}
              importance={headImportance}
              tokens={inputSequence}
              activePos={currentPos}
              {highlightedTarget}
              on:headHover={handleHeadHover}
              on:headSelect={(e) => {
                activeSeveredHead = e.detail.head;
                showTryAnother = false;
              }}
            />
          {/if}
        </div>
      </div>
    {/if}
  </main>

  <!-- Chapter compass: persistent 'you are here' rail on wide screens -->
  {#if !isLoading && !loadError}
    <nav
      class="hidden min-[1440px]:flex fixed left-4 top-1/2 -translate-y-1/2 flex-col items-center gap-1 z-40"
      aria-label="Page chapters"
    >
      {#each [
        { n: 1, label: 'Give it a story start' },
        { n: 2, label: 'Watch it think' },
        { n: 3, label: 'Act 1 · Look inside' },
        { n: 4, label: 'Act 2 · Break it' },
      ] as ch}
        <a
          href={`#chapter-${ch.n}`}
          title={ch.label}
          class={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-mono font-bold transition-all duration-300 ${
            activeChapter === ch.n
              ? 'bg-ink text-white scale-110 shadow-sm'
              : 'bg-white text-ink-muted border border-hairline hover:border-ink hover:text-ink'
          }`}
        >
          {ch.n}
        </a>
        {#if ch.n < 4}<span class="w-px h-5 bg-hairline"></span>{/if}
      {/each}
    </nav>
  {/if}

  <!-- 5-slide guided tour: one concept at a time, with simple diagrams -->
  <GuidedTour open={tourOpen} on:close={() => (tourOpen = false)} on:done={() => (tourOpen = false)} />

  <!-- High-Contrast Technical Telemetry Footer -->
  <footer class="border-t border-hairline py-3 px-4 sm:px-6 bg-white text-xs font-mono">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2 min-w-0">
      <span class="text-slate-800 font-bold">15.2M Parameters · 6 Layers · 6 Heads · d=288</span>
      <span class="text-slate-900 font-extrabold">Token {currentToken} · Position {currentPos}</span>
    </div>
  </footer>
</div>
