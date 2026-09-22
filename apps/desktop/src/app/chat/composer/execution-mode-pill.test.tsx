import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { ExecutionModePill } from './execution-mode-pill'

const view = { $runtimeId: atom('A'), $busy: atom(false), $model: atom('local'), $provider: atom('custom') }
const request = vi.hoisted(() => vi.fn())
vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => view }))
vi.mock('@/store/session-states', () => ({
  knownOwnerForSession: (id: string) => ({ profile: id }),
  requestForOwnedSession: request
}))
vi.mock('@/components/ui/tooltip', () => ({ Tip: ({ children }: React.PropsWithChildren) => children }))
vi.mock('@/components/ui/dropdown-menu', () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => children,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => children,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => children,
  DropdownMenuRadioGroup: ({ children, value, onValueChange }: React.PropsWithChildren<{
    value: string, onValueChange: (value: string) => void
  }>) => <select aria-label="Mode" value={value} onChange={e => onValueChange(e.target.value)}>{children}</select>,
  DropdownMenuRadioItem: ({ children, value }: React.PropsWithChildren<{ value: string }>) =>
    <option value={value}>{children}</option>
}))

afterEach(() => { cleanup(); request.mockReset(); view.$runtimeId.set('A'); view.$busy.set(false) })

it('discards a response belonging to the previously selected session', async () => {
  let resolveA!: (result: unknown) => void
  request.mockImplementation((id: string) => id === 'A'
    ? new Promise(resolve => { resolveA = resolve })
    : Promise.resolve({ available: true, policy: 'auto', active: 'agent' }))
  render(<ExecutionModePill disabled={false} />)
  act(() => view.$runtimeId.set('B'))
  expect(await screen.findByLabelText('Auto · Agent')).toBeTruthy()
  await act(async () => resolveA({ available: true, policy: 'chat', active: 'chat' }))
  expect(screen.queryByLabelText('Chat')).toBeNull()
  expect(screen.getByLabelText('Auto · Agent')).toBeTruthy()
})

it('queues a mode on the owning backend and hides unsupported sessions', async () => {
  request.mockResolvedValueOnce({ available: true, policy: 'auto', active: 'chat' })
    .mockResolvedValueOnce({ available: true, policy: 'auto', active: 'chat', requested: 'agent' })
    .mockResolvedValueOnce({ available: false, policy: 'agent', active: 'agent' })
  render(<ExecutionModePill disabled={false} />)
  await screen.findByLabelText('Auto · Chat')
  fireEvent.change(screen.getByLabelText('Mode'), { target: { value: 'agent' } })
  await screen.findByLabelText('Agent')
  expect(request.mock.calls[1][0]).toBe('A')
  expect(request.mock.calls[1][3]).toEqual({ session_id: 'A', mode: 'agent' })
  act(() => view.$runtimeId.set('B'))
  await waitFor(() => expect(screen.queryByLabelText('Mode')).toBeNull())
})
