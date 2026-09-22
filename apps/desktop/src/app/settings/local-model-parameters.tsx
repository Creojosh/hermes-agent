import { useEffect, useState } from 'react'

import { getLocalModelSettings, type ModelSettingsResponse, saveLocalModelSettings } from '@/api/local-models'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'

interface LocalModelParametersProps {
  modelId: string
  onClose: () => void
  onSaved: () => void
}

export function LocalModelParameters({ modelId, onClose, onSaved }: LocalModelParametersProps) {
  const { t } = useI18n()
  const copy = t.settings.localModels.parameters
  const [data, setData] = useState<ModelSettingsResponse | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    let cancelled = false
    void getLocalModelSettings(modelId)
      .then(result => {
        if (!cancelled) {
          setData(result)
          setValues(result.values)
        }
      })
      .catch(reason => {
        if (!cancelled) {
          setError(String(reason))
        }
      })

    return () => {
      cancelled = true
    }
  }, [modelId])

  const save = async () => {
    setSaving(true)
    setError('')

    try {
      await saveLocalModelSettings(modelId, values)
      onSaved()
      onClose()
    } catch (reason) {
      setError(String(reason))
    } finally {
      setSaving(false)
    }
  }

  const setValue = (key: string, value: string) =>
    setValues(current => {
      const next = { ...current }

      if (value) {
        next[key] = value
      } else {
        delete next[key]
      }

      return next
    })

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !saving) {
          onClose()
        }
      }}
      open
    >
      <DialogContent bodyClassName="flex min-h-0 flex-col" className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription className="break-all">{modelId}</DialogDescription>
        </DialogHeader>
        <p className="text-xs text-muted-foreground">{copy.detail}</p>
        <SearchField aria-label={copy.search} onChange={setQuery} placeholder={copy.search} value={query} />
        {!data && !error && <Loader2 className="size-4 animate-spin" />}
        <div className="min-h-0 max-h-[55vh] overflow-y-auto space-y-6 pr-2">
          {(['gpu', 'context', 'cache', 'sampling', 'penalties', 'rope', 'speculative', 'vision'] as const).map(
            group => {
              const fields =
                data?.fields.filter(
                  field =>
                    field.group === group &&
                    `${field.key} ${copy[group]}`.toLowerCase().includes(query.trim().toLowerCase())
                ) ?? []

              if (!fields.length) {
                return null
              }

              return (
                <section className="space-y-3" key={group}>
                  <h3 className="text-sm font-medium">{copy[group]}</h3>
                  <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
                    {fields.map(field => (
                      <div className="space-y-1.5" key={field.key}>
                        <label className="block text-xs text-muted-foreground" htmlFor={`model-${field.key}`}>
                          {field.key}
                        </label>
                        {field.choices ? (
                          <Select
                            disabled={saving}
                            onValueChange={value => setValue(field.key, value === '__inherit' ? '' : value)}
                            value={values[field.key] || '__inherit'}
                          >
                            <SelectTrigger aria-label={field.key} id={`model-${field.key}`}>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="__inherit">
                                {copy.inherit}
                                {data?.inherited[field.key] ? ` · ${data.inherited[field.key]}` : ''}
                              </SelectItem>
                              {field.choices.map(value => (
                                <SelectItem key={value} value={value}>
                                  {value}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        ) : (
                          <Input
                            aria-label={field.key}
                            disabled={saving}
                            id={`model-${field.key}`}
                            max={field.kind === 'probability' ? 1 : undefined}
                            min={field.min ?? undefined}
                            onChange={event => setValue(field.key, event.target.value)}
                            placeholder={
                              data?.inherited[field.key] || (field.key === 'tensor-split' ? '2,1' : copy.inherit)
                            }
                            step={field.kind === 'integer' ? 1 : 'any'}
                            type={field.kind === 'text' ? 'text' : 'number'}
                            value={values[field.key] ?? ''}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )
            }
          )}
        </div>
        {error && (
          <p className="text-xs text-destructive" role="alert">
            {error}
          </p>
        )}
        <p className="text-xs text-muted-foreground">{copy.restart}</p>
        <DialogFooter>
          <Button disabled={saving || !data} onClick={() => setValues({})} variant="ghost">
            {copy.reset}
          </Button>
          <Button disabled={saving || !data} onClick={() => void save()}>
            {saving && <Loader2 className="animate-spin" />}
            {copy.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
