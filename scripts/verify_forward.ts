import fs from 'node:fs';
import path from 'node:path';
import { parseCheckpoint } from '../src/engine/loader';
import { parseTokenizer } from '../src/engine/tokenizer';
import { forward, createRunBuffers } from '../src/engine/forward';
import { HookRegistry } from '../src/engine/hooks';

async function downloadFile(url: string, destPath: string) {
  if (fs.existsSync(destPath)) {
    console.log(`Using cached file: ${destPath}`);
    return;
  }
  console.log(`Downloading ${url} -> ${destPath}...`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download ${url}: ${res.statusText}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  fs.writeFileSync(destPath, Buffer.from(arrayBuffer));
  console.log(`Downloaded ${destPath} (${(arrayBuffer.byteLength / 1024 / 1024).toFixed(2)} MB)`);
}

async function main() {
  const cacheDir = path.join(process.cwd(), '.cache');
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  const modelPath = path.join(cacheDir, 'stories15M.bin');
  const tokenizerPath = path.join(cacheDir, 'tokenizer.bin');

  await downloadFile(
    'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin',
    modelPath
  );
  await downloadFile(
    'https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin',
    tokenizerPath
  );

  console.log('Loading checkpoint...');
  const modelBuf = fs.readFileSync(modelPath);
  // Ensure arrayBuffer slice
  const modelArrayBuf = modelBuf.buffer.slice(
    modelBuf.byteOffset,
    modelBuf.byteOffset + modelBuf.byteLength
  );
  const { config, weights } = parseCheckpoint(modelArrayBuf);
  console.log('Model config loaded:', config);

  console.log('Loading tokenizer...');
  const tokBuf = fs.readFileSync(tokenizerPath);
  const tokArrayBuf = tokBuf.buffer.slice(
    tokBuf.byteOffset,
    tokBuf.byteOffset + tokBuf.byteLength
  );
  const tokenizer = parseTokenizer(tokArrayBuf, config.vocabSize);
  console.log('Tokenizer loaded. Vocab size:', tokenizer.vocabSize);

  // Initialize buffers & hooks
  const buffers = createRunBuffers(config);
  const hooks = new HookRegistry();

  // Test token: BOS = 1 (standard beginning-of-sequence in Llama-2 / llama2.c)
  const inputToken = 1;
  const pos = 0;

  console.log(`\nRunning single forward pass for token ${inputToken} ("${tokenizer.vocab[inputToken]}") at pos ${pos}...`);
  const start = performance.now();
  const logits = forward(inputToken, pos, config, weights, buffers, hooks);
  const elapsed = (performance.now() - start).toFixed(2);
  console.log(`Forward pass completed in ${elapsed} ms.`);

  // Find top 5 tokens
  const logitsData = logits.data.subarray(logits.offset, logits.offset + config.vocabSize);
  const indexedLogits = Array.from(logitsData).map((logit, idx) => ({ idx, logit }));
  indexedLogits.sort((a, b) => b.logit - a.logit);

  console.log('\nTop 5 tokens:');
  for (let i = 0; i < 5; i++) {
    const item = indexedLogits[i];
    const piece = tokenizer.decode(inputToken, item.idx);
    console.log(`  #${i + 1}: token_id=${item.idx} ("${piece}") logit=${item.logit.toFixed(4)}`);
  }

  // Sanity assertions: a numerically broken forward pass must fail this script
  const allFinite = Array.from(logitsData).every((v) => Number.isFinite(v));
  if (!allFinite) {
    throw new Error('NaN or Infinity detected in logits — forward pass is numerically broken');
  }
  const topLogit = indexedLogits[0].logit;
  const secondLogit = indexedLogits[1].logit;
  if (topLogit <= secondLogit || topLogit <= 0) {
    throw new Error(`Degenerate logits: top=${topLogit.toFixed(4)} second=${secondLogit.toFixed(4)}`);
  }
  // stories15M convention: after BOS, common story openers dominate the top
  const topPiece = tokenizer.decode(inputToken, indexedLogits[0].idx);
  if (!topPiece || topPiece.trim().length === 0) {
    throw new Error(`Top prediction decoded to empty string (id=${indexedLogits[0].idx})`);
  }
  console.log(`\n✓ Sanity checks passed — logits finite, top="${topPiece}", spread=${(topLogit - secondLogit).toFixed(4)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
