import { test, expect, type Page } from '@playwright/test';

// Regression for the prefill bug: App pre-seeded inputSequence with the whole
// prompt, so advanceGeneration's prefill branch never ran and the controller's
// KV cache received only BOS + whatever argmax emitted. argmax-after-BOS is
// "Once", which masked the bug on the default prompt — but the mirror ran four
// tokens ahead of the real sequence and any non-default prompt was ignored.
//
// Observable invariant: the tape renders exactly one chip per fed position,
// so chip[pos].id must equal the expected token at that position, and
// chips.length must equal footer Position + 1 (mirror == controller).

// Ground truth from scripts/precompute_scans.ts (argmax continuation,
// MAX_GENERATED=16, verified against the live engine by the drift test).
const EXPECTED: Record<string, { chips: number[]; promptIds: number[]; pos: number; token: number }> = {
  'Once upon a time': {
    promptIds: [9038, 2501, 263, 931],
    chips: [1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708, 5377, 297],
    pos: 20,
    token: 297,
  },
  'The little girl saw a bird': {
    promptIds: [450, 2217, 7826, 4446, 263, 11199],
    chips: [1, 450, 2217, 7826, 4446, 263, 11199, 297, 278, 14744, 29889, 2296, 471, 577, 24173, 322, 5131, 304, 4380, 372, 29889, 2296, 6350],
    pos: 22,
    token: 6350,
  },
  'A dragon wakes': {
    promptIds: [319, 8338, 265, 281, 6926],
    chips: [1, 319, 8338, 265, 281, 6926, 297, 278, 13569, 29889, 739, 338, 1407, 11220, 322, 278, 8805, 338, 13031, 292, 29889, 450],
    pos: 21,
    token: 450,
  },
};

async function chipTable(page: Page) {
  return page.$$eval('[title^="Position #"]', (els) =>
    els
      .map((el) => {
        const t = el.getAttribute('title') ?? '';
        return {
          pos: Number(t.match(/#(\d+)/)?.[1]),
          id: Number(t.match(/ID (\d+)/)?.[1]),
          kind: t.includes('your prompt token')
            ? 'prompt'
            : t.includes('written by the model')
              ? 'generated'
              : 'bos',
        };
      })
      .sort((a, b) => a.pos - b.pos)
  );
}

async function playToEnd(page: Page) {
  const ctrl = page.locator('#panel-controls');
  await ctrl.getByRole('button', { name: 'Step' }).waitFor({ state: 'visible', timeout: 30_000 });
  // 1x -> 2x -> 4x (160ms per step)
  const speedBtn = ctrl.getByTitle('Play speed (click to cycle)');
  await speedBtn.click();
  await speedBtn.click();
  await ctrl.getByRole('button', { name: 'Play', exact: true }).click();
  await ctrl.getByRole('button', { name: 'Play', exact: true }).waitFor({ state: 'visible', timeout: 300_000 });
}

test.describe('prefill regression: the prompt actually enters the model', () => {
  test('default prompt: fed sequence is BOS + prompt + argmax, run ends at pos 20', async ({ page }) => {
    const exp = EXPECTED['Once upon a time'];
    await page.goto('/');
    await playToEnd(page);

    const footer = await page.locator('footer').textContent();
    expect(footer).toContain(`Token ${exp.token} · Position ${exp.pos}`);

    const chips = await chipTable(page);
    expect(chips.map((c) => c.id), 'chip ids = exact fed sequence').toEqual(exp.chips);

    // Mirror == controller: one chip per position, positions 0..pos.
    expect(chips.length).toBe(exp.pos + 1);
    expect(chips.map((c) => c.pos)).toEqual(exp.chips.map((_, i) => i));

    // Prompt appears exactly once, as prompt chips — never as generated ones.
    const promptChips = chips.filter((c) => c.kind === 'prompt');
    expect(promptChips.map((c) => c.id)).toEqual(exp.promptIds);
    expect(chips.filter((c) => c.kind === 'generated').length).toBe(exp.pos - exp.promptIds.length);
  });

  test('bird preset: generated story is conditioned on the actual prompt', async ({ page }) => {
    const exp = EXPECTED['The little girl saw a bird'];
    await page.goto('/');
    await page.getByRole('button', { name: /The little girl/ }).click();
    await playToEnd(page);

    const footer = await page.locator('footer').textContent();
    expect(footer).toContain(`Token ${exp.token} · Position ${exp.pos}`);

    const chips = await chipTable(page);
    expect(chips.map((c) => c.id)).toEqual(exp.chips);
    expect(chips.length).toBe(exp.pos + 1);

    const promptChips = chips.filter((c) => c.kind === 'prompt');
    expect(promptChips.map((c) => c.id)).toEqual(exp.promptIds);
    // Under the bug, the first generated chips re-spelled "Once upon a time".
    expect(chips.filter((c) => c.kind === 'generated').slice(0, 4).map((c) => c.id)).not.toEqual(
      [9038, 2501, 263, 931]
    );
  });

  test('typed custom prompt: fed sequence is BOS + typed tokens + argmax', async ({ page }) => {
    const exp = EXPECTED['A dragon wakes'];
    await page.goto('/');
    await page.locator('#panel-controls').getByRole('button', { name: 'Step' }).waitFor({
      state: 'visible',
      timeout: 30_000,
    });
    await page.locator('#prompt-input').fill('A dragon wakes');
    await playToEnd(page);

    const footer = await page.locator('footer').textContent();
    expect(footer).toContain(`Token ${exp.token} · Position ${exp.pos}`);

    const chips = await chipTable(page);
    expect(chips.map((c) => c.id), 'chip ids = exact fed sequence').toEqual(exp.chips);
    expect(chips.length).toBe(exp.pos + 1);

    const promptChips = chips.filter((c) => c.kind === 'prompt');
    expect(promptChips.map((c) => c.id)).toEqual(exp.promptIds);
    // The typed prompt must be what entered the model — nothing else.
    expect(chips.filter((c) => c.kind === 'generated').slice(0, 4).map((c) => c.id)).not.toEqual(
      [9038, 2501, 263, 931]
    );
  });
});
