import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup,
  DropdownMenuRadioItem, DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'

interface ExecutionMode {
  available: boolean
  policy: 'auto' | 'chat' | 'agent'
  active: 'chat' | 'agent'
  requested?: 'auto' | 'chat' | 'agent' | null
}

const LABELS = { auto: 'Auto', chat: 'Chat', agent: 'Agent' }
const rejectUnowned = async <T,>(): Promise<T> => { throw new Error('Session owner unavailable') }

interface ExecutionModePillProps { disabled: boolean }

/** Backend-owned mode, read on session/turn boundaries; no classifier in the renderer. */
export function ExecutionModePill({ disabled }: ExecutionModePillProps) {
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  const busy = useStore(view.$busy)
  const model = useStore(view.$model)
  const provider = useStore(view.$provider)
  const [snapshot, setSnapshot] = useState<ExecutionMode | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const generation = useRef(0)
  useEffect(() => {
    const mine = ++generation.current
    setSnapshot(null)
    setError('')
    setSaving(false)
    if (!sessionId || busy) return
    const owner = JSON.stringify(knownOwnerForSession(sessionId))
    void requestForOwnedSession<ExecutionMode>(sessionId, rejectUnowned, 'session.execution_mode', {
      session_id: sessionId
    }).then(result => {
      if (generation.current === mine && owner === JSON.stringify(knownOwnerForSession(sessionId))) setSnapshot(result)
    }).catch(() => { /* Older backends do not expose this optional control. */ })
    return () => { generation.current++ }
  }, [sessionId, busy, model, provider])

  if (!snapshot?.available || !sessionId) return null

  const selected = snapshot.requested ?? snapshot.policy
  const label = snapshot.requested ? LABELS[snapshot.requested]
    : snapshot.policy === 'auto' ? `Auto · ${LABELS[snapshot.active]}` : LABELS[snapshot.active]
  const save = async (mode: string) => {
    const mine = ++generation.current
    const owner = JSON.stringify(knownOwnerForSession(sessionId))
    setSaving(true)
    setError('')
    try {
      const result = await requestForOwnedSession<ExecutionMode>(sessionId, rejectUnowned, 'session.execution_mode', {
        session_id: sessionId, mode
      })
      if (generation.current === mine && owner === JSON.stringify(knownOwnerForSession(sessionId))) setSnapshot(result)
    } catch (failure) {
      if (generation.current === mine) setError(String(failure))
    } finally {
      if (generation.current === mine) setSaving(false)
    }
  }

  return <DropdownMenu>
    <Tip label={error || label}>
      <DropdownMenuTrigger asChild>
        <Button aria-label={label} disabled={disabled || busy || saving} variant="ghost"
          className="h-(--composer-control-size) shrink-0 rounded-md px-2 text-xs font-normal text-(--ui-text-tertiary)">
          {error ? `${label} !` : label}
        </Button>
      </DropdownMenuTrigger>
    </Tip>
    <DropdownMenuContent>
      <DropdownMenuRadioGroup value={selected} onValueChange={mode => void save(mode)}>
        {Object.entries(LABELS).map(([mode, text]) =>
          <DropdownMenuRadioItem key={mode} value={mode}>{text}</DropdownMenuRadioItem>)}
      </DropdownMenuRadioGroup>
    </DropdownMenuContent>
  </DropdownMenu>
}
