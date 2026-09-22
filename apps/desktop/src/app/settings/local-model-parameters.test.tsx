import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import { getLocalModelSettings, saveLocalModelSettings } from '@/api/local-models'
import { I18nProvider } from '@/i18n'

import { LocalModelParameters } from './local-model-parameters'

vi.mock('@/api/local-models', () => ({ getLocalModelSettings: vi.fn(), saveLocalModelSettings: vi.fn() }))

beforeEach(() => {
  vi.mocked(getLocalModelSettings).mockResolvedValue({
    values: { 'ctx-size': '8192' },
    inherited: {},
    fields: [{ key: 'ctx-size', group: 'context', kind: 'integer', min: 512, choices: null }]
  })
  vi.mocked(saveLocalModelSettings).mockResolvedValue({ ok: true })
})

it('saves only the selected model and keeps failed drafts editable', async () => {
  const close = vi.fn()
  vi.mocked(saveLocalModelSettings).mockRejectedValueOnce(new Error('Server unavailable'))
  render(
    <I18nProvider>
      <LocalModelParameters modelId="Alpha" onClose={close} onSaved={vi.fn()} />
    </I18nProvider>
  )
  const input = await screen.findByRole('spinbutton', { name: 'ctx-size' })
  fireEvent.change(input, { target: { value: '16384' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save parameters' }))
  expect((await screen.findByRole('alert')).textContent).toContain('Server unavailable')
  expect(close).not.toHaveBeenCalled()
  expect((input as HTMLInputElement).value).toBe('16384')
  fireEvent.click(screen.getByRole('button', { name: 'Save parameters' }))
  await waitFor(() => expect(close).toHaveBeenCalledOnce())
  expect(saveLocalModelSettings).toHaveBeenLastCalledWith('Alpha', { 'ctx-size': '16384' })
})

it('finds advanced groups and preserves hidden edits when filtering and saving', async () => {
  vi.mocked(getLocalModelSettings).mockResolvedValue({
    values: { 'ctx-size': '8192' },
    inherited: {},
    fields: [
      { key: 'ctx-size', group: 'context', kind: 'integer', min: 512, choices: null },
      { key: 'presence-penalty', group: 'penalties', kind: 'number', min: null, choices: null },
      { key: 'rope-scale', group: 'rope', kind: 'positive', min: 0, choices: null },
      { key: 'spec-draft-n-max', group: 'speculative', kind: 'integer', min: 0, choices: null },
      { key: 'image-max-tokens', group: 'vision', kind: 'integer', min: -1, choices: null }
    ]
  })
  render(
    <I18nProvider>
      <LocalModelParameters modelId="Alpha" onClose={vi.fn()} onSaved={vi.fn()} />
    </I18nProvider>
  )
  fireEvent.change(await screen.findByRole('spinbutton', { name: 'presence-penalty' }), { target: { value: '-0.5' } })
  const search = screen.getByRole('textbox', { name: 'Search parameters…' })
  fireEvent.change(search, { target: { value: ' YaRN ' } })
  expect(screen.queryByRole('spinbutton', { name: 'presence-penalty' })).toBeNull()
  fireEvent.change(screen.getByRole('spinbutton', { name: 'rope-scale' }), { target: { value: '2' } })
  fireEvent.change(search, { target: { value: '' } })
  fireEvent.change(screen.getByRole('spinbutton', { name: 'spec-draft-n-max' }), { target: { value: '3' } })
  fireEvent.change(screen.getByRole('spinbutton', { name: 'image-max-tokens' }), { target: { value: '1024' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save parameters' }))
  await waitFor(() =>
    expect(saveLocalModelSettings).toHaveBeenCalledWith('Alpha', {
      'ctx-size': '8192',
      'presence-penalty': '-0.5',
      'rope-scale': '2',
      'spec-draft-n-max': '3',
      'image-max-tokens': '1024'
    })
  )
})
