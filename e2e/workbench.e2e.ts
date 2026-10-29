import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';

test.describe('workbench smoke test', () => {
  test('loads the full page: hero, controls, token tape, pipeline, panels', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Execution Pipeline')).toBeVisible();
    await expect(page.getByText('Attention Field')).toBeVisible();
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
});
