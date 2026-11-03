import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';

test.describe('workbench smoke test', () => {
  test('loads the full page: hero, controls, token tape, pipeline, panels', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Execution Pipeline')).toBeVisible();
    await expect(page.getByText('Attention Field', { exact: true })).toBeVisible();
    // Act 2 starts collapsed behind the teaser card
    await expect(page.getByText('Now try breaking it')).toBeVisible();
    await page.getByText('Now try breaking it').click();
    await expect(page.getByText('Circuit Map', { exact: true })).toBeVisible();
    // Token tape starts with the BOS chip
    await expect(page.getByText('⟨s⟩').first()).toBeVisible();
    // Narrative bar opens with the prompt's first token queued
    await expect(page.getByText(/Token \d+ Queued at Position 0/)).toBeVisible();
  });

  test('preset chips fill the prompt and populate the token tape', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Once upon/ }).click();
    await expect(page.getByText('Once').first()).toBeVisible();
    await expect(page.getByText('time').first()).toBeVisible();
  });

  test('Step advances the model and the narrative follows it', async ({ page }) => {
    await page.goto('/');
    const step = page.getByRole('button', { name: 'Step' });
    await expect(step).toBeEnabled({ timeout: 30_000 });
    await step.click();
    // After one step the QUEUED tag must be replaced by a live phase tag
    await expect(page.getByText(/QUEUED/).first()).not.toBeVisible();
    await expect(page.getByText(/LAYER \d|ATTENTION|FFN|PREDICTION/i).first()).toBeVisible();
  });

  test('no critical accessibility violations', async ({ page }) => {
    await page.goto('/');
    // Wait for the app to be fully mounted before scanning
    await expect(page.getByText('Execution Pipeline')).toBeVisible();
    const results = await new AxeBuilder({ page }).analyze();
    const critical = results.violations.filter((v) => v.impact === 'critical');
    expect(
      critical.map((v) => `${v.id} (${v.nodes.length} nodes)`),
      'critical axe violations found'
    ).toEqual([]);
  });

  test('390px: act switcher swaps panels and nothing overflows', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await expect(page.getByText('Working Vector')).toBeVisible({ timeout: 30_000 });

    // Mobile: only Act 1 visible by default, switcher present
    await expect(page.getByRole('button', { name: 'Act 2 · Experiment' })).toBeVisible();
    await expect(page.getByText('Circuit Map', { exact: true })).not.toBeVisible();

    // Switch to Act 2 (teaser card first, then the real panel)
    await page.getByRole('button', { name: 'Act 2 · Experiment' }).click();
    await expect(page.getByText('Now try breaking it')).toBeVisible();
    await page.getByText('Now try breaking it').click();
    await expect(page.getByText('Circuit Map', { exact: true })).toBeVisible();
    await expect(page.getByText('Working Vector')).not.toBeVisible();

    const overflow = await page.evaluate(() => {
      const docOverflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
      const overflowingElements: string[] = [];
      for (const el of Array.from(document.querySelectorAll('header, main, footer, section, main > div > *'))) {
        const rect = el.getBoundingClientRect();
        if (rect.right > window.innerWidth + 0.5 || rect.left < -0.5) {
          overflowingElements.push(`${el.tagName}.${el.className}: right=${rect.right.toFixed(1)}`);
        }
      }
      return { docOverflow, overflowingElements };
    });

    expect(overflow.overflowingElements).toEqual([]);
    expect(overflow.docOverflow).toBeLessThanOrEqual(0);
  });

  test('1280px: both acts side by side, no switcher, no overflow', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/');
    await expect(page.getByText('Working Vector')).toBeVisible({ timeout: 30_000 });
    // Act 2 opens beside Act 1 on desktop once the teaser is clicked
    await page.getByText('Now try breaking it').click();
    await expect(page.getByText('Circuit Map', { exact: true })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole('button', { name: 'Act 2 · Experiment' })).not.toBeVisible();

    // Guided 3-step strip: step 2 runs the curated head for the default prompt
    await page.getByRole('button', { name: /^0:0/ }).click();
    await expect(page.getByText('Head L0:H0 Switched Off')).toBeVisible();
    await page.getByRole('button', { name: /2\.\s*Switch off L0:H1/ }).click();
    await expect(page.getByText('Head L0:H1 Switched Off')).toBeVisible();
    await expect(page.locator('#panel-circuit-map')).toHaveClass(/ring-amber-400/);
    await expect(page.getByText('Now try another square.')).toBeVisible();

    const docOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth
    );
    expect(docOverflow).toBeLessThanOrEqual(0);
  });
});
