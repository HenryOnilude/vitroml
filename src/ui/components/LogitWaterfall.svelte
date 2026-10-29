<script lang="ts">
  import type { TokenPrediction } from '../../engine/analysis';

  export let ablatedTop5: TokenPrediction[] = [];
  export let baselineTop5: TokenPrediction[] = [];

  $: maxAbsDelta = Math.max(
    ...ablatedTop5.map((t) => Math.abs(t.delta ?? 0)),
    0.2
  );
</script>

<div class="space-y-1.5 font-mono text-xs">
  <div class="text-[10px] font-sans font-medium text-slate-900">
    Top-5 guesses: score before → after switching the head off
  </div>
  {#each ablatedTop5 as item, idx}
    {@const delta = item.delta ?? 0}
    {@const isPos = delta >= 0}
    {@const barWidthPct = Math.min(100, (Math.abs(delta) / maxAbsDelta) * 100)}
    {@const baselineItem = baselineTop5.find((b) => b.tokenId === item.tokenId)}

    <div class="bg-white border border-hairline rounded p-2 hover:border-hairline-bright transition flex flex-col gap-1.5 shadow-subtle">
      <!-- Top Row: Rank, Token String, Token ID, Logit Values -->
      <div class="flex items-center justify-between text-[11px]">
        <div class="flex items-center gap-1.5">
          <span class="text-ink-muted text-[10px] w-3 font-semibold">#{idx + 1}</span>
          <span class="text-ink font-bold font-sans">{item.tokenPiece}</span>
          <span class="text-[9px] text-ink-muted">[{item.tokenId}]</span>
        </div>

        <div class="flex items-center gap-2">
          {#if baselineItem}
            <span class="text-ink-muted text-[10px]">{baselineItem.logit.toFixed(2)} →</span>
          {/if}
          <span class="text-ink font-bold">{item.logit.toFixed(2)}</span>
          <span
            class="text-[10px] px-1.5 py-0.2 rounded font-semibold border"
            style={isPos
              ? 'color: #8c4a22; background: #f7ece2; border-color: #e5cdb8'
              : 'color: #1f5a7d; background: #e6eff5; border-color: #c3d6e2'}
          >
            {isPos ? '+' : ''}{delta.toFixed(3)}
          </span>
        </div>
      </div>

      <!-- Bottom Row: Diverging Delta Bar (Zero in Center) -->
      <div class="relative h-1 bg-canvas-elevated rounded-full overflow-hidden flex items-center">
        <!-- Center line -->
        <div class="absolute left-1/2 top-0 bottom-0 w-[1px] bg-hairline-bright z-10"></div>

        {#if isPos}
          <!-- Positive bar extending right from 50% (vik positive ochre) -->
          <div
            class="h-full rounded-r-full transition-all duration-200 shadow-sm"
            style={`position: absolute; left: 50%; width: ${barWidthPct / 2}%; background: #c37243;`}
          ></div>
        {:else}
          <!-- Negative bar extending left from 50% (vik negative navy) -->
          <div
            class="h-full rounded-l-full transition-all duration-200 shadow-sm"
            style={`position: absolute; right: 50%; width: ${barWidthPct / 2}%; background: #307da6;`}
          ></div>
        {/if}
      </div>
    </div>
  {/each}
</div>
