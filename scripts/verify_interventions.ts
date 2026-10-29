import fs from 'node:fs';
import path from 'node:path';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { forward, createRunBuffers } from '../src/engine/forward';
import { HookRegistry } from '../src/engine/hooks';
import { zeroAblateHead } from '../src/engine/interventions';

async function main() {
  const cacheDir = path.join(process.cwd(), '.cache');
  const modelPath = path.join(cacheDir, 'stories15M.bin');
  const tokenizerPath = path.join(cacheDir, 'tokenizer.bin');

  if (!fs.existsSync(modelPath) || !fs.existsSync(tokenizerPath)) {
    throw new Error('Model or tokenizer not found in .cache. Run npm run verify first.');
  }

  console.log('Loading checkpoint and tokenizer...');
  const modelBuf = fs.readFileSync(modelPath);
  const modelArrayBuf = modelBuf.buffer.slice(
    modelBuf.byteOffset,
    modelBuf.byteOffset + modelBuf.byteLength
  );
  const { config, weights } = parseCheckpoint(modelArrayBuf);

  const tokBuf = fs.readFileSync(tokenizerPath);
  const tokArrayBuf = tokBuf.buffer.slice(
    tokBuf.byteOffset,
    tokBuf.byteOffset + tokBuf.byteLength
  );
  const tokenizer = parseTokenizer(tokArrayBuf, config.vocabSize);

  const inputToken = 1; // BOS
  const pos = 0;

  // 1. Baseline Run
  console.log('\n--- 1. Baseline Forward Pass ---');
  const baselineBuffers = createRunBuffers(config);
  const baselineHooks = new HookRegistry();
  const baselineLogitsTensor = forward(inputToken, pos, config, weights, baselineBuffers, baselineHooks);
  const baselineLogits = Array.from(
    baselineLogitsTensor.data.subarray(
      baselineLogitsTensor.offset,
      baselineLogitsTensor.offset + config.vocabSize
    )
  );

  const baselineSorted = baselineLogits
    .map((logit, idx) => ({ idx, logit }))
    .sort((a, b) => b.logit - a.logit);

  console.log('Top 5 baseline predictions:');
  for (let i = 0; i < 5; i++) {
    const item = baselineSorted[i];
    console.log(`  #${i + 1}: [${item.idx}] "${tokenizer.decode(inputToken, item.idx)}" = ${item.logit.toFixed(4)}`);
  }

  // 2. Intervened Run: Zero ablate head 0 in layer 0
  const targetLayer = 0;
  const targetHead = 0;
  console.log(`\n--- 2. Intervened Forward Pass (Zero Ablate Layer ${targetLayer}, Head ${targetHead}) ---`);
  const ablatedBuffers = createRunBuffers(config);
  const ablatedHooks = new HookRegistry();
  zeroAblateHead(ablatedHooks, targetLayer, targetHead, config.nHeads, config.dim);

  const ablatedLogitsTensor = forward(inputToken, pos, config, weights, ablatedBuffers, ablatedHooks);
  const ablatedLogits = Array.from(
    ablatedLogitsTensor.data.subarray(
      ablatedLogitsTensor.offset,
      ablatedLogitsTensor.offset + config.vocabSize
    )
  );

  const ablatedSorted = ablatedLogits
    .map((logit, idx) => ({ idx, logit }))
    .sort((a, b) => b.logit - a.logit);

  console.log('Top 5 ablated predictions:');
  for (let i = 0; i < 5; i++) {
    const item = ablatedSorted[i];
    console.log(`  #${i + 1}: [${item.idx}] "${tokenizer.decode(inputToken, item.idx)}" = ${item.logit.toFixed(4)}`);
  }

  // 3. Compute logit difference statistics
  let maxAbsDiff = 0.0;
  let sumSquaredDiff = 0.0;
  for (let i = 0; i < config.vocabSize; i++) {
    const diff = ablatedLogits[i] - baselineLogits[i];
    const absDiff = Math.abs(diff);
    if (absDiff > maxAbsDiff) {
      maxAbsDiff = absDiff;
    }
    sumSquaredDiff += diff * diff;
  }
  const l2Diff = Math.sqrt(sumSquaredDiff);

  console.log('\n--- Intervention Impact Summary ---');
  console.log(`L2 norm of logit difference: ${l2Diff.toFixed(4)}`);
  console.log(`Max absolute logit difference: ${maxAbsDiff.toFixed(4)}`);

  console.log('\nLogit changes on top baseline tokens:');
  for (let i = 0; i < 5; i++) {
    const idx = baselineSorted[i].idx;
    const piece = tokenizer.decode(inputToken, idx);
    const bVal = baselineLogits[idx];
    const aVal = ablatedLogits[idx];
    const delta = aVal - bVal;
    console.log(
      `  Token [${idx}] "${piece}": baseline=${bVal.toFixed(4)} -> ablated=${aVal.toFixed(4)} (delta: ${delta > 0 ? '+' : ''}${delta.toFixed(4)})`
    );
  }

  if (maxAbsDiff === 0.0) {
    throw new Error('Intervention had zero effect on logits! Verification failed.');
  } else {
    console.log('\nSUCCESS: Intervention verified! The output logits changed meaningfully.');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
