import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { NEW_CHAT_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import {
  activateLocalModel,
  configureLocalModelsDirectory,
  configureLocalRuntime,
  deleteLocalModel,
  downloadBrowsedModel,
  downloadLocalModel,
  ejectLocalModel,
  getLocalCatalog,
  getLocalHardware,
  getLocalModelsStatus,
  getLocalRuntimeCapabilities,
  type HFFileGroup,
  type HFSearchHit,
  listHFRepoFiles,
  quickstartLocalModels,
  searchHFModels,
  setLocalModelVision,
  setLocalServer,
  sideloadLocalModel
} from '@/hermes'
import { useI18n } from '@/i18n'
import {
  Check,
  CheckCircle2,
  Cpu,
  Download,
  Eject,
  FolderOpen,
  Loader2,
  Monitor,
  Package,
  Search,
  StopFilled,
  Trash2,
  Zap
} from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $localRuntimeInstallStarting,
  $localRuntimeJobs,
  runningDownloadFor,
  runningRuntimeInstall,
  startLocalRuntimeInstall,
  watchLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import { notify, notifyError } from '@/store/notifications'
import type {
  LocalCatalogModel,
  LocalHardware,
  LocalModelsStatus,
  LocalRuntimeCapabilities,
  LocalRuntimeOption
} from '@/types/hermes'

import { LocalModelParameters } from './local-model-parameters'
import { ListRow, Pill, SettingsContent, SettingsSection, SettingsSkeleton } from './primitives'
import { ActiveProfileNote } from './profile-scope'

function ProgressBar({ percent }: { percent: number | undefined }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
      <div
        className="h-full rounded-full bg-primary transition-[width] duration-300"
        style={{ width: `${Math.max(2, Math.min(100, percent ?? 2))}%` }}
      />
    </div>
  )
}

function gbLabel(bytes: number | null | undefined): string {
  if (!bytes) {
    return '—'
  }

  return `${(bytes / (1 << 30)).toFixed(1)} GB`
}

// Catalog display order: what runs well leads. Resident (all on GPU)
// first, then spilled (works, slower), then doesn't-fit; catalog order
// (recommended first) holds within each band.
function fitRank(model: LocalCatalogModel): number {
  if (model.fits && !model.spilled) {
    return 0
  }

  if (model.fits) {
    return 1
  }

  return 2
}

type RuntimeSelections = Record<string, boolean | string>

function runtimeOptionFlag(option: LocalRuntimeOption): string {
  return option.flags.find(flag => flag.startsWith('--')) ?? option.flags[0]
}

function usableRuntimeOptions(options: LocalRuntimeOption[]): LocalRuntimeOption[] {
  const seen = new Set<string>()
  const usable: LocalRuntimeOption[] = []

  for (const option of options) {
    const flags = option.flags.filter(flag => /^--?[a-zA-Z0-9]/.test(flag))

    if (!flags.length || flags.some(flag => ['--help', '--version', '--cache-list', '--completion-bash'].includes(flag))) {
      continue
    }

    const cleaned = { ...option, flags }
    const canonical = runtimeOptionFlag(cleaned)

    if (!seen.has(canonical)) {
      seen.add(canonical)
      usable.push(cleaned)
    }
  }

  return usable
}

function decodeRuntimeArgs(args: string[], options: LocalRuntimeOption[]) {
  const aliases = new Map(options.flatMap(option => option.flags.map(flag => [flag, option] as const)))
  const selected: RuntimeSelections = {}
  const unknown: string[] = []

  for (let index = 0; index < args.length; index += 1) {
    const option = aliases.get(args[index])

    if (!option) {
      unknown.push(args[index])

      continue
    }

    const flag = runtimeOptionFlag(option)

    if (option.value) {
      selected[flag] = args[index + 1] ?? ''
      index += 1
    } else {
      selected[flag] = true
    }
  }

  return { selected, unknown }
}

function encodeRuntimeArgs(options: LocalRuntimeOption[], selected: RuntimeSelections, unknown: string[]): string[] {
  const args = [...unknown]

  for (const option of options) {
    const flag = runtimeOptionFlag(option)

    if (!(flag in selected)) {
      continue
    }

    if (option.value) {
      const value = typeof selected[flag] === 'string' ? selected[flag].trim() : ''

      if (!value) {
        continue
      }

      args.push(flag, value)
    } else {
      args.push(flag)
    }
  }

  return args
}

export function LocalModelsSettings() {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const installStarting = useStore($localRuntimeInstallStarting)
  const [status, setStatus] = useState<LocalModelsStatus | null>(null)
  const [hardware, setHardware] = useState<LocalHardware | null>(null)
  const [catalog, setCatalog] = useState<LocalCatalogModel[] | null>(null)
  const [deleting, setDeleting] = useState<null | string>(null)
  const [modelsDirectorySaving, setModelsDirectorySaving] = useState(false)
  const [visionSaving, setVisionSaving] = useState<null | string>(null)
  const [parametersModel, setParametersModel] = useState<string | null>(null)
  const [serverBusy, setServerBusy] = useState(false)
  // Advanced configuration is the front door: no model or runtime choice is imposed.
  const [configure, setConfigure] = useState(true)
  const [runtimeCapabilities, setRuntimeCapabilities] = useState<LocalRuntimeCapabilities | null>(null)
  const [runtimePathDraft, setRuntimePathDraft] = useState('')
  const [runtimeSelections, setRuntimeSelections] = useState<RuntimeSelections>({})
  const [runtimeUnknownArgs, setRuntimeUnknownArgs] = useState<string[]>([])
  const [runtimeOptionQuery, setRuntimeOptionQuery] = useState('')
  const [runtimeSaving, setRuntimeSaving] = useState(false)
  const [runtimeDraftInitialized, setRuntimeDraftInitialized] = useState(false)
  // Jobs live in the app-level store (they must survive this pane
  // unmounting); the pane just renders the slice it cares about.
  const jobs = useStore($localRuntimeJobs)

  const refresh = useCallback(() => {
    void getLocalModelsStatus()
      .then(setStatus)
      .catch(() => setStatus(null))
    void getLocalCatalog()
      .then(data => setCatalog(data.models))
      .catch(() => setCatalog([]))
    void getLocalRuntimeCapabilities()
      .then(setRuntimeCapabilities)
      .catch(() => setRuntimeCapabilities(null))
  }, [])

  // Snappy first paint: status + catalog immediately; hardware (may shell out
  // to nvidia-smi) backfills and pops in-place. The job watcher also kicks
  // here so reopening the pane rediscovers work started before.
  useEffect(() => {
    refresh()
    watchLocalRuntimeJobs()
    void getLocalHardware()
      .then(setHardware)
      .catch(() => setHardware(null))
  }, [refresh])

  useEffect(() => {
    if (status && runtimeCapabilities && !runtimeDraftInitialized) {
      setRuntimeDraftInitialized(true)
      setRuntimePathDraft(status.runtime_path || '')
      const decoded = decodeRuntimeArgs(status.runtime_args || [], usableRuntimeOptions(runtimeCapabilities.options))
      setRuntimeSelections(decoded.selected)
      setRuntimeUnknownArgs(decoded.unknown)
    }
  }, [runtimeCapabilities, runtimeDraftInitialized, status])

  // The pane is LIVE while visible: residency changes without user action
  // (boot warm finishing, idle sweep unloading, another surface ejecting),
  // and a stale snapshot here reads as a broken feature — 'VRAM full but
  // the pane says Not in memory'. The status route is built cheap for
  // polling; setTimeout chain, never overlapping.
  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const tick = async () => {
      try {
        const next = await getLocalModelsStatus()

        if (!cancelled) {
          setStatus(next)
        }
      } catch {
        // Backend briefly unreachable — keep the last snapshot.
      }

      if (!cancelled) {
        timer = window.setTimeout(() => void tick(), 4_000)
      }
    }

    timer = window.setTimeout(() => void tick(), 4_000)

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [])

  // A job finishing (download done, install done) changes what status/catalog
  // should show — refresh whenever the running set shrinks.
  const runningCount = jobs.filter(j => j.status === 'running').length
  useEffect(() => {
    refresh()
  }, [refresh, runningCount])

  async function handleQuickstart() {
    try {
      await quickstartLocalModels()
      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.quickstartFailed)
    }
  }

  async function handleDownload(model: LocalCatalogModel) {
    try {
      const res = await downloadLocalModel(model.id)

      if (res.already_downloaded || !res.job_id) {
        refresh()

        return
      }

      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.downloadFailed(model.display_name))
    }
  }

  async function saveModelsDirectory(path: string) {
    setModelsDirectorySaving(true)

    try {
      const result = await configureLocalModelsDirectory(path)
      notify({
        durationMs: 3_500,
        kind: 'success',
        message: copy.modelsDirectorySaved(result.detected_models),
        title: copy.title
      })
      refresh()
    } catch (err) {
      notifyError(err, copy.modelsDirectoryFailed)
    } finally {
      setModelsDirectorySaving(false)
    }
  }

  async function chooseModelsDirectory() {
    try {
      const paths = await window.hermesDesktop.selectPaths({
        directories: true,
        multiple: false,
        title: copy.modelsDirectoryChoose
      })

      if (paths[0]) {
        await saveModelsDirectory(paths[0])
      }
    } catch (err) {
      notifyError(err, copy.modelsDirectoryFailed)
    }
  }

  async function handleActivate(target: null | string, displayName: string) {
    if (!target) {
      return
    }

    try {
      await activateLocalModel(target)
      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.activateFailed(displayName))
    }
  }

  async function handleEject(modelId: string) {
    try {
      await ejectLocalModel(modelId)
      notify({ durationMs: 3_000, kind: 'success', message: copy.ejected, title: copy.title })
      refresh()
    } catch (err) {
      notifyError(err, copy.ejectFailed)
    }
  }

  async function handleVision(modelId: string, enabled: boolean) {
    setVisionSaving(modelId)

    try {
      await setLocalModelVision(modelId, enabled)
      refresh()
    } catch (err) {
      notifyError(err, copy.visionFailed)
    } finally {
      setVisionSaving(null)
    }
  }

  async function handleServer(action: 'start' | 'stop') {
    setServerBusy(true)

    try {
      await setLocalServer(action)
      notify({
        durationMs: 3_500,
        kind: 'success',
        message: action === 'stop' ? copy.serverStopped : copy.serverStarted,
        title: copy.title
      })
      refresh()
    } catch (err) {
      notifyError(err, action === 'stop' ? copy.serverStopFailed : copy.serverStartFailed)
    } finally {
      setServerBusy(false)
    }
  }

  async function chooseRuntimeFolder() {
    try {
      const paths = await window.hermesDesktop.selectPaths({
        directories: true,
        multiple: false,
        title: 'Choose the folder containing llama-server'
      })

      if (paths[0]) {
        setRuntimePathDraft(paths[0])

        const args = encodeRuntimeArgs(
          usableRuntimeOptions(runtimeCapabilities?.options ?? []),
          runtimeSelections,
          runtimeUnknownArgs
        )

        const capabilities = await getLocalRuntimeCapabilities(paths[0])
        const decoded = decodeRuntimeArgs(args, usableRuntimeOptions(capabilities.options))
        setRuntimeCapabilities(capabilities)
        setRuntimeSelections(decoded.selected)
        setRuntimeUnknownArgs(decoded.unknown)
      }
    } catch (err) {
      notifyError(err, 'Could not choose the runtime folder')
    }
  }

  async function saveRuntimeConfiguration() {
    setRuntimeSaving(true)

    try {
      const extraArgs = encodeRuntimeArgs(
        usableRuntimeOptions(runtimeCapabilities?.options ?? []),
        runtimeSelections,
        runtimeUnknownArgs
      )

      const result = await configureLocalRuntime(runtimePathDraft, extraArgs)
      setRuntimeCapabilities(result.capabilities)
      notify({ durationMs: 3_000, kind: 'success', message: 'Runtime configuration saved.', title: copy.title })
      refresh()
    } catch (err) {
      notifyError(err, 'Could not save the runtime configuration')
    } finally {
      setRuntimeSaving(false)
    }
  }

  function toggleRuntimeOption(option: LocalRuntimeOption, enabled: boolean) {
    const flag = runtimeOptionFlag(option)
    setRuntimeSelections(current => {
      const next = { ...current }

      if (enabled) {
        next[flag] = option.value ? '' : true
      } else {
        delete next[flag]
      }

      return next
    })
  }

  async function handleDelete(target: string, rowId: string) {
    if (!window.confirm(copy.deleteConfirm(target))) {
      return
    }

    setDeleting(rowId)

    try {
      await deleteLocalModel(target)
      notify({ durationMs: 2_500, kind: 'success', message: copy.deleted(target), title: copy.title })
      refresh()
    } catch (err) {
      notifyError(err, copy.deleteFailed)
    } finally {
      setDeleting(null)
    }
  }

  // Setup flows end at the action, not the settings pane: when quickstart
  // finishes while the user is still HERE watching it, land them on a new
  // chat with the model ready to try. Unmount cancels the intent — a user
  // who navigated away mid-download keeps their place (no focus theft).
  // (Lives above the loading return: hooks run unconditionally.)
  const navigate = useNavigate()
  const seenQuickstarts = useRef(new Set<string>())

  const runningQuickstart = jobs.find(j => j.kind === 'quickstart' && j.status === 'running')

  useEffect(() => {
    // Event detection, not value mirroring: the ref only remembers which
    // job ids THIS mount saw running, so a 'done' already in the list on
    // mount (stale history) never triggers a navigation.
    const seen = seenQuickstarts.current

    for (const j of jobs) {
      if (j.kind !== 'quickstart') {
        continue
      }

      if (j.status === 'running') {
        seen.add(j.job_id)
      } else if (j.status === 'done' && seen.has(j.job_id)) {
        seen.delete(j.job_id)
        navigate(NEW_CHAT_ROUTE)
      }
    }
  }, [jobs, navigate])

  if (!status || catalog === null) {
    return <SettingsSkeleton sections={[{ rows: 2 }, { rows: 4 }]} />
  }

  const rJob = runningRuntimeInstall(jobs)
  const lastError = jobs.find(j => j.status === 'error')

  const sortedCatalog = catalog
    .filter(model => !status.models_dir_custom || model.downloaded)
    .sort((a, b) => fitRank(a) - fitRank(b))

  const detectedRuntimeOptions = usableRuntimeOptions(runtimeCapabilities?.options ?? [])

  const visibleRuntimeOptions = detectedRuntimeOptions.filter(option => {
    const haystack = `${option.flags.join(' ')} ${option.value} ${option.description}`.toLowerCase()

    return haystack.includes(runtimeOptionQuery.trim().toLowerCase())
  })

  const runtimeHasIncompleteValues = detectedRuntimeOptions.some(option => {
    const flag = runtimeOptionFlag(option)

    return option.value && flag in runtimeSelections && !String(runtimeSelections[flag]).trim()
  })

  // ── Quickstart: the dummy-proof front door ──
  // Until something is servable (runtime + at least one model), the pane
  // leads with a hero that does everything in one click; the full pane
  // stays one 'Let me choose' click away. A running quickstart pins this
  // view so its progress has a home even after a remount.
  const qJob = runningQuickstart ?? null

  const needsSetup = !status.runtime_installed || status.models.length === 0
  // The setup hero is reserved for an automatic recommendation. A
  // spilled model remains visible below, but setup must not silently choose it.
  const heroModel = status.models_dir_custom ? null : (catalog.find(c => c.recommended && c.fits) ?? null)
  const hasRecommendation = !status.models_dir_custom && catalog.some(c => c.recommended)

  const failedInstall = jobs.some(job => job.kind === 'runtime-install' && job.status === 'error')

  if (qJob || (needsSetup && !configure && heroModel && !installStarting && !rJob && !failedInstall)) {
    // Stage rail derived from the job phase: engine -> model -> finish.
    const phase = qJob?.phase ?? ''

    const stageIndex = ['starting-server', 'setting-default'].includes(phase) ? 2 : phase === 'downloading' ? 1 : 0

    const stages = [copy.quickstartStageEngine, copy.quickstartStageModel, copy.quickstartStageFinish]

    // The model-download leg blanks job.detail on purpose (pane rows
    // render their own byte counter) — compose one here instead of
    // falling back to runtime copy that would misname the stage.
    const liveDetail =
      qJob &&
      (qJob.detail ||
        (qJob.total_bytes
          ? copy.downloadProgress(gbLabel(qJob.done_bytes), gbLabel(qJob.total_bytes))
          : copy.installing))

    return (
      <SettingsContent>
        <div className="flex min-h-[60dvh] items-center justify-center">
          <div className="w-full max-w-md text-center">
            <div className="mx-auto mb-5 flex size-14 items-center justify-center rounded-2xl bg-primary/10">
              {qJob ? (
                <Loader2 className="size-7 animate-spin text-primary" />
              ) : (
                <Cpu className="size-7 text-primary" />
              )}
            </div>

            <h2 className="text-lg font-semibold text-foreground">
              {qJob ? qJob.target : (heroModel?.display_name ?? '')}
            </h2>

            {qJob ? (
              <>
                <p className="mt-2 min-h-10 text-[0.8rem] leading-5 text-muted-foreground">{liveDetail}</p>

                <div className="mt-5">
                  <ProgressBar percent={qJob.percent} />
                </div>

                {/* Stage rail: engine -> model -> finish. */}
                <div className="mt-5 flex items-center justify-center gap-5">
                  {stages.map((label, i) => (
                    <span
                      className={cn(
                        'inline-flex items-center gap-1.5 text-[0.72rem]',
                        i < stageIndex && 'text-(--ui-text-tertiary)',
                        i === stageIndex && 'font-medium text-foreground',
                        i > stageIndex && 'text-(--ui-text-tertiary) opacity-60'
                      )}
                      key={label}
                    >
                      {i < stageIndex ? (
                        <CheckCircle2 className="size-3.5 text-primary" />
                      ) : i === stageIndex ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <span className="size-1.5 rounded-full bg-current" />
                      )}
                      {label}
                    </span>
                  ))}
                </div>
              </>
            ) : heroModel ? (
              <>
                <p className="mt-2 text-[0.8rem] leading-5 text-muted-foreground">
                  {heroModel.downloaded
                    ? copy.quickstartDetailReady(heroModel.display_name)
                    : copy.quickstartDetail(heroModel.display_name, heroModel.size_label)}
                </p>

                <div className="mt-6 flex items-center justify-center gap-3">
                  <Button onClick={() => setConfigure(true)} size="sm" variant="outline">
                    {copy.quickstartConfigure}
                  </Button>
                  <Button onClick={() => void handleQuickstart()} size="default">
                    <Zap />
                    {copy.quickstartAction}
                  </Button>
                </div>
              </>
            ) : null}

            {lastError?.kind === 'quickstart' && !qJob && (
              <p className="mt-4 text-[0.75rem] text-destructive">{lastError.error}</p>
            )}
          </div>
        </div>
      </SettingsContent>
    )
  }

  // Up to date = the authority (status) says the configured tag is what's
  // serving. Shown whenever true — not only right after an update.
  const updateApplied = status.runtime_installed && !status.update_available && status.tag === status.configured_tag

  return (
    <SettingsContent>
      <ActiveProfileNote className="mb-5" />
      {/* ── Runtime ── */}
      <SettingsSection
        aside={
          status.runtime_installed ? (
            <Pill tone="primary">
              {status.server_running ? copy.serverRunning : copy.runtimeReady(status.runtime_backend ?? '')}
            </Pill>
          ) : undefined
        }
        icon={Zap}
        meta={status.tag}
        title={copy.runtimeTitle}
      >
        {status.runtime_installed ? (
          <ListRow
            action={
              status.server_running ? (
                <Button
                  className={cn(serverBusy && '[&_svg]:animate-spin')}
                  disabled={serverBusy}
                  onClick={() => void handleServer('stop')}
                  size="sm"
                  variant="outline"
                >
                  {serverBusy ? <Loader2 /> : <StopFilled />}
                  {copy.stopServer}
                </Button>
              ) : (
                <Button
                  className={cn(serverBusy && '[&_svg]:animate-spin')}
                  disabled={serverBusy}
                  onClick={() => void handleServer('start')}
                  size="sm"
                  variant="outline"
                >
                  {serverBusy ? <Loader2 /> : <Zap />}
                  {copy.startServer}
                </Button>
              )
            }
            description={
              status.server_running
                ? copy.runtimeRunningDetail
                : copy.runtimeInstalledDetail(status.tag, status.runtime_backend ?? 'cpu')
            }
            title={copy.runtimeInstalled}
          />
        ) : rJob ? (
          <ListRow
            below={<ProgressBar percent={rJob.percent} />}
            description={rJob.detail || copy.installing}
            title={
              <span className="inline-flex items-center gap-2">
                <Loader2 className="size-3.5 animate-spin" />
                {copy.installing}
              </span>
            }
          />
        ) : (
          <ListRow
            action={
              <Button disabled={installStarting} onClick={() => void startLocalRuntimeInstall()} size="sm">
                <Download />
                {copy.installAction}
              </Button>
            }
            description={copy.installDetail}
            title={copy.installTitle}
          />
        )}

        {status.update_available && !rJob && (
          <ListRow
            action={
              <Button disabled={installStarting} onClick={() => void startLocalRuntimeInstall()} size="sm">
                <Download />
                {copy.updateAction}
              </Button>
            }
            description={copy.updateDetail(status.configured_tag, status.tag)}
            title={copy.updateTitle}
          />
        )}

        {rJob && status.runtime_installed && (
          <ListRow
            below={<ProgressBar percent={rJob.percent} />}
            description={rJob.detail || copy.updating}
            title={
              <span className="inline-flex items-center gap-2">
                <Loader2 className="size-3.5 animate-spin" />
                {copy.updating}
              </span>
            }
          />
        )}

        {updateApplied && (
          <ListRow
            description={copy.upToDateDetail(status.tag, status.runtime_backend ?? 'cpu')}
            title={
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 dark:text-emerald-400" />
                {copy.upToDateTitle}
              </span>
            }
          />
        )}

        {lastError?.kind === 'runtime-install' && <p className="text-[0.75rem] text-destructive">{lastError.error}</p>}

        <ListRow
          action={
            <div className="flex items-center gap-2">
              {runtimePathDraft && (
                <Button onClick={() => setRuntimePathDraft('')} size="sm" variant="ghost">
                  Use managed
                </Button>
              )}
              <Button onClick={() => void chooseRuntimeFolder()} size="sm" variant="outline">
                <FolderOpen />
                Choose folder
              </Button>
            </div>
          }
          description={runtimePathDraft || 'Hermes-managed llama.cpp runtime'}
          title="Runtime directory"
        />

        <details>
          <summary className="cursor-pointer text-sm font-medium">{copy.parameters.global}</summary>
          <p className="mt-2 text-xs text-muted-foreground">{copy.parameters.globalDetail}</p>
          <ListRow
            action={
              <Button
                disabled={runtimeSaving || runtimeHasIncompleteValues}
                onClick={() => void saveRuntimeConfiguration()}
                size="sm"
              >
                {runtimeSaving ? <Loader2 className="animate-spin" /> : <Check />}
                Save
              </Button>
            }
            below={
              runtimeCapabilities?.executable ? (
                <div className="mt-2 grid gap-2">
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      aria-label="Search llama.cpp options"
                      className="pl-8"
                      onChange={event => setRuntimeOptionQuery(event.target.value)}
                      placeholder="Search options…"
                      value={runtimeOptionQuery}
                    />
                  </div>

                  <div className="max-h-80 overflow-auto rounded-md border border-(--ui-border)">
                    {visibleRuntimeOptions.map(option => {
                      const flag = runtimeOptionFlag(option)
                      const enabled = flag in runtimeSelections
                      const value = typeof runtimeSelections[flag] === 'string' ? runtimeSelections[flag] : ''

                      return (
                        <label
                          className="grid cursor-pointer grid-cols-[auto_minmax(0,1fr)_minmax(8rem,12rem)] items-center gap-2 border-b border-(--ui-border) px-2.5 py-2 last:border-b-0 hover:bg-(--ui-bg-tertiary)"
                          key={flag}
                        >
                          <input
                            checked={enabled}
                            className="size-3.5 accent-primary"
                            onChange={event => toggleRuntimeOption(option, event.target.checked)}
                            type="checkbox"
                          />
                          <span className="min-w-0">
                            <code className="text-[0.72rem] font-medium text-foreground">
                              {option.flags.join(', ')}
                            </code>
                            {option.description && (
                              <span className="mt-0.5 block text-[0.68rem] leading-4 text-muted-foreground">
                                {option.description}
                              </span>
                            )}
                          </span>
                          {option.value ? (
                            <Input
                              aria-label={`Value for ${flag}`}
                              disabled={!enabled}
                              onChange={event =>
                                setRuntimeSelections(current => ({ ...current, [flag]: event.target.value }))
                              }
                              onClick={event => event.stopPropagation()}
                              placeholder={option.value}
                              size="sm"
                              value={value}
                            />
                          ) : (
                            <span className="text-right text-[0.68rem] text-muted-foreground">
                              {enabled ? 'Enabled' : 'Disabled'}
                            </span>
                          )}
                        </label>
                      )
                    })}

                    {visibleRuntimeOptions.length === 0 && (
                      <p className="px-3 py-5 text-center text-xs text-muted-foreground">No matching option.</p>
                    )}
                  </div>

                  {runtimeUnknownArgs.length > 0 && (
                    <div className="text-xs text-muted-foreground">
                      <code className="block break-all">{JSON.stringify(runtimeUnknownArgs)}</code>
                      <Button onClick={() => setRuntimeUnknownArgs([])} size="sm" variant="ghost">
                        {copy.parameters.removeUnknown}
                      </Button>
                      <p>
                        {runtimeUnknownArgs.length} argument(s) not exposed by this runtime are preserved unchanged.
                      </p>
                    </div>
                  )}
                  {runtimeHasIncompleteValues && (
                    <p className="text-[0.68rem] text-destructive">Enter a value for every enabled option.</p>
                  )}
                </div>
              ) : (
                <p className="mt-2 text-xs text-muted-foreground">
                  Install a runtime or choose a llama.cpp folder to detect its available controls.
                </p>
              )
            }
            description="Enable an option, then enter only its value. Controls come directly from llama-server --help."
            title="llama.cpp options"
          />
        </details>

        {runtimeCapabilities?.executable && (
          <details className="rounded-md border border-(--ui-border) px-3 py-2 text-xs">
            <summary className="cursor-pointer font-medium">
              Raw llama-server --help ({detectedRuntimeOptions.length} detected options)
            </summary>
            <p className="mt-2 break-all text-muted-foreground">
              {runtimeCapabilities.executable}
              {runtimeCapabilities.version ? ` · ${runtimeCapabilities.version}` : ''}
            </p>
            <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-(--ui-bg-tertiary) p-2 font-mono text-[0.68rem]">
              {runtimeCapabilities.help_text}
            </pre>
          </details>
        )}
      </SettingsSection>

      {/* ── This machine ── */}
      <SettingsSection icon={Monitor} title={copy.hardwareTitle}>
        {hardware ? (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
            {(hardware.gpus?.length ? hardware.gpus : hardware.gpu_name ? [{ name: hardware.gpu_name }] : []).map(
              (gpu, index) => (
                <span className="inline-flex items-center gap-1.5" key={`${gpu.name}-${index}`}>
                  <Zap className="size-3.5" />
                  {gpu.name}
                </span>
              )
            )}

            <span className="inline-flex items-center gap-1.5">
              <Cpu className="size-3.5" />
              {copy.vram(gbLabel(hardware.vram_total_bytes))}
            </span>

            <span className="inline-flex items-center gap-1.5">
              <Package className="size-3.5" />
              {copy.ram(gbLabel(hardware.ram_total_bytes))}
            </span>

            {hardware.uma && <Pill>{copy.unifiedMemory}</Pill>}
          </div>
        ) : (
          <p className="py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
            {copy.hardwareLoading}
          </p>
        )}
      </SettingsSection>

      {/* ── Models ── */}
      <SettingsSection
        icon={Download}
        meta={`${status.models_dir_custom ? status.models.length : catalog.length}`}
        title={copy.modelsTitle}
      >
        {status.models_dir_custom !== undefined && (
          <ListRow
            action={
              <div className="flex items-center gap-2">
                {status.models_dir_custom && (
                  <Button
                    disabled={modelsDirectorySaving}
                    onClick={() => void saveModelsDirectory('')}
                    size="sm"
                    variant="ghost"
                  >
                    {copy.modelsDirectoryDefault}
                  </Button>
                )}
                <Button
                  disabled={modelsDirectorySaving}
                  onClick={() => void chooseModelsDirectory()}
                  size="sm"
                  variant="outline"
                >
                  {modelsDirectorySaving ? <Loader2 className="animate-spin" /> : <FolderOpen />}
                  {copy.modelsDirectoryAction}
                </Button>
              </div>
            }
            description={copy.modelsDirectoryDescription(status.models_dir)}
            title={copy.modelsDirectoryTitle}
          />
        )}

        {!status.models_dir_custom && !hasRecommendation && (
          <ListRow
            action={
              <Button
                onClick={() =>
                  document.getElementById('local-model-browse')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }
                size="sm"
              >
                <Search />
                {copy.noRecommendationAction}
              </Button>
            }
            description={copy.noRecommendationDetail}
            title={copy.noRecommendationTitle}
          />
        )}

        <div className="grid gap-1">
          {sortedCatalog.map(model => {
            const dJob = runningDownloadFor(jobs, model.id)
            const anyDownloadRunning = jobs.some(j => j.kind === 'model-download' && j.status === 'running')
            const activateTarget = model.downloaded_model_id ?? model.model_id
            const stagedModel = status.models.find(item => item.id === activateTarget)
            const isActive = Boolean(activateTarget && status.active_model_id === activateTarget)
            const residency = activateTarget ? status.loaded_models[activateTarget] : undefined
            const isLoaded = residency === 'loaded' || residency === 'ready'
            const isLoadingNow = residency === 'loading'
            const livePlacement = activateTarget ? status.placement?.[activateTarget] : undefined

            const aJob = jobs.find(
              j => j.kind === 'model-activate' && j.status === 'running' && j.model_id === activateTarget
            )

            const anyActivateRunning = jobs.some(j => j.kind === 'model-activate' && j.status === 'running')

            return (
              <ListRow
                action={
                  model.downloaded ? (
                    <div className="flex items-center justify-end gap-2">
                      {activateTarget && (
                        <Button onClick={() => setParametersModel(activateTarget)} size="sm" variant="outline">
                          {copy.parameters.title}
                        </Button>
                      )}
                      {stagedModel?.vision_available && (
                        <label className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                          <input
                            aria-label={`${copy.visionToggle} ${model.display_name}`}
                            checked={Boolean(stagedModel.vision_enabled)}
                            disabled={visionSaving === stagedModel.id}
                            onChange={event => void handleVision(stagedModel.id, event.target.checked)}
                            type="checkbox"
                          />
                          {copy.visionToggle}
                        </label>
                      )}
                      {isLoaded && livePlacement && (
                        <Tip label={livePlacement.spilled ? copy.placementSpilledTip : copy.placementResidentTip}>
                          <Pill tone={livePlacement.spilled ? 'warn' : 'success'}>
                            <Cpu className="mr-1 size-3" />
                            {livePlacement.granted_window_label ?? livePlacement.window_label ?? ''}
                            {' · '}
                            {livePlacement.spilled ? copy.placementSpilled : copy.placementResident}
                          </Pill>
                        </Tip>
                      )}
                      {isLoaded && !livePlacement && <Pill>{copy.loadedPill}</Pill>}

                      {isLoadingNow && (
                        <Pill>
                          <Loader2 className="mr-1 size-3 animate-spin" />
                          {copy.loadingPill}
                        </Pill>
                      )}

                      {isActive ? (
                        <Tip label={copy.activeDetail}>
                          <Pill tone="primary">
                            <Check className="mr-1 size-3" />
                            {copy.activePill}
                          </Pill>
                        </Tip>
                      ) : (
                        <Button
                          className={cn(aJob && '[&_svg]:animate-spin')}
                          disabled={anyActivateRunning}
                          onClick={() => void handleActivate(activateTarget ?? null, model.display_name)}
                          size="sm"
                        >
                          {aJob ? <Loader2 /> : <Check />}
                          {aJob ? copy.activating : copy.useAction}
                        </Button>
                      )}

                      {isLoaded && (
                        <Tip label={copy.ejectTip}>
                          <Button
                            onClick={() => void handleEject(activateTarget ?? model.id)}
                            size="icon"
                            variant="ghost"
                          >
                            <Eject />
                          </Button>
                        </Tip>
                      )}

                      <Tip label={copy.deleteAction}>
                        <Button
                          className={cn(deleting === model.id && '[&_svg]:animate-spin')}
                          onClick={() => void handleDelete(model.downloaded_model_id ?? model.id, model.id)}
                          size="icon"
                          variant="ghost"
                        >
                          {deleting === model.id ? <Loader2 /> : <Trash2 />}
                        </Button>
                      </Tip>
                    </div>
                  ) : dJob ? undefined : (
                    <Button
                      disabled={anyDownloadRunning}
                      onClick={() => void handleDownload(model)}
                      size="sm"
                      variant="outline"
                    >
                      <Download />
                      {copy.downloadAction(model.size_label)}
                    </Button>
                  )
                }
                below={
                  dJob ? (
                    <div className="mt-2 grid gap-1">
                      <ProgressBar percent={dJob.percent} />

                      <p className="text-[0.68rem] text-muted-foreground">
                        {!dJob.done_bytes && dJob.detail
                          ? dJob.detail
                          : copy.downloadProgress(gbLabel(dJob.done_bytes), gbLabel(dJob.total_bytes))}
                      </p>
                    </div>
                  ) : undefined
                }
                description={
                  <>
                    {model.description}

                    <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      {/* Memory: the traffic light. Green = runs fully on
                          the GPU; amber = spills to system RAM (works,
                          slower); red = doesn't fit this machine at all.
                          Detail prose lives in the tooltip. */}
                      {!model.fits ? (
                        <Tip label={model.fit_detail ?? model.fit_summary}>
                          <Pill tone="destructive">
                            <Cpu className="mr-1 size-3" />
                            {copy.pillTooBig}
                          </Pill>
                        </Tip>
                      ) : model.spilled ? (
                        <Tip label={model.quant_reason ?? model.fit_summary}>
                          <Pill tone="warn">
                            <Cpu className="mr-1 size-3" />
                            {copy.pillUsesRam}
                          </Pill>
                        </Tip>
                      ) : (
                        <Tip label={model.quant_reason ?? model.fit_summary}>
                          <Pill tone="success">
                            <Cpu className="mr-1 size-3" />
                            {copy.pillFitsGpu}
                          </Pill>
                        </Tip>
                      )}

                      {/* Context: one pill. Green 'Full X context' only when
                          the model earned its complete window resident on the
                          GPU — a big context served from system RAM is slow,
                          and a green badge there would sell exactly the wrong
                          model, so a spilled full window goes gray. Anything
                          starting below its native window gets one quiet
                          'Up to' pill instead of a start/grow pair. */}
                      {model.fits &&
                        model.start_window_label &&
                        (model.start_window && model.start_window >= model.native_context ? (
                          <Tip label={copy.pillFullContextTip}>
                            <Pill tone={model.spilled ? 'muted' : 'success'}>
                              {copy.pillFullContext(model.native_context_label)}
                            </Pill>
                          </Tip>
                        ) : (
                          <Tip label={copy.pillGrowsTip}>
                            <Pill>{copy.pillUpTo(model.native_context_label)}</Pill>
                          </Tip>
                        ))}

                      {!model.fits && <Pill>{copy.pillUpTo(model.native_context_label)}</Pill>}
                    </span>

                    {isActive && !isLoaded && !isLoadingNow && status.server_running && (
                      <span className="mt-0.5 block text-(--ui-text-tertiary)">{copy.activeNotLoaded}</span>
                    )}
                  </>
                }
                key={model.id}
                title={
                  <span className="inline-flex items-center gap-2">
                    {model.display_name}

                    {!status.models_dir_custom &&
                      model.recommended &&
                      (model.recommended_reason ? (
                        // The why, straight from the resolver: the tooltip is
                        // the branch that picked this model, so the shown
                        // rationale can never drift from the actual decision.
                        <Tip label={copy.recommendedReason[model.recommended_reason]}>
                          <Pill tone="primary">{copy.recommended}</Pill>
                        </Tip>
                      ) : (
                        <Pill tone="primary">{copy.recommended}</Pill>
                      ))}
                  </span>
                }
              />
            )
          })}

          {status.models
            .filter(m => !catalog.some(c => c.downloaded_model_id === m.id || c.model_id === m.id))
            .map(m => {
              const isActive = status.active_model_id === m.id
              const residency = status.loaded_models[m.id]
              const isLoaded = residency === 'loaded' || residency === 'ready'
              const isLoadingNow = residency === 'loading'
              const livePlacement = status.placement?.[m.id]

              const aJob = jobs.find(j => j.kind === 'model-activate' && j.status === 'running' && j.model_id === m.id)

              const anyActivateRunning = jobs.some(j => j.kind === 'model-activate' && j.status === 'running')

              return (
                <ListRow
                  action={
                    <div className="flex items-center justify-end gap-2">
                      <Button onClick={() => setParametersModel(m.id)} size="sm" variant="outline">
                        {copy.parameters.title}
                      </Button>
                      {m.vision_available && (
                        <label className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                          <input
                            aria-label={`${copy.visionToggle} ${m.id}`}
                            checked={Boolean(m.vision_enabled)}
                            disabled={visionSaving === m.id}
                            onChange={event => void handleVision(m.id, event.target.checked)}
                            type="checkbox"
                          />
                          {copy.visionToggle}
                        </label>
                      )}
                      {isLoaded && livePlacement && (
                        <Tip label={livePlacement.spilled ? copy.placementSpilledTip : copy.placementResidentTip}>
                          <Pill tone={livePlacement.spilled ? 'warn' : 'success'}>
                            <Cpu className="mr-1 size-3" />
                            {livePlacement.granted_window_label ?? livePlacement.window_label ?? ''}
                            {' · '}
                            {livePlacement.spilled ? copy.placementSpilled : copy.placementResident}
                          </Pill>
                        </Tip>
                      )}
                      {isLoaded && !livePlacement && <Pill>{copy.loadedPill}</Pill>}

                      {isLoadingNow && (
                        <Pill>
                          <Loader2 className="mr-1 size-3 animate-spin" />
                          {copy.loadingPill}
                        </Pill>
                      )}

                      {isActive ? (
                        <Pill tone="primary">
                          <CheckCircle2 className="mr-1 size-3" />
                          {copy.activePill}
                        </Pill>
                      ) : (
                        <Button
                          className={cn(aJob && '[&_svg]:animate-spin')}
                          disabled={anyActivateRunning}
                          onClick={() => void handleActivate(m.id, m.id)}
                          size="sm"
                        >
                          {aJob ? <Loader2 /> : <Check />}
                          {copy.useAction}
                        </Button>
                      )}

                      {isLoaded && (
                        <Tip label={copy.ejectTip}>
                          <Button onClick={() => void handleEject(m.id)} size="icon" variant="ghost">
                            <Eject />
                          </Button>
                        </Tip>
                      )}

                      <Tip label={copy.deleteAction}>
                        <Button
                          className={cn(deleting === m.id && '[&_svg]:animate-spin')}
                          onClick={() => void handleDelete(m.id, m.id)}
                          size="icon"
                          variant="ghost"
                        >
                          {deleting === m.id ? <Loader2 /> : <Trash2 />}
                        </Button>
                      </Tip>
                    </div>
                  }
                  description={<span>{copy.addedByYou}</span>}
                  key={m.id}
                  title={
                    <span className="inline-flex items-center gap-2">
                      <span className="truncate font-mono text-[0.8rem]">{m.id}</span>

                      <span className="text-[0.68rem] font-normal text-muted-foreground">{m.size_label}</span>
                    </span>
                  }
                />
              )
            })}
        </div>

        {lastError?.kind === 'model-download' && <p className="text-[0.75rem] text-destructive">{lastError.error}</p>}
      </SettingsSection>

      <BrowseSection onChanged={refresh} />
      {parametersModel && (
        <LocalModelParameters
          key={parametersModel}
          modelId={parametersModel}
          onClose={() => setParametersModel(null)}
          onSaved={refresh}
        />
      )}
    </SettingsContent>
  )
}

function fitTone(fit: HFFileGroup['fit']): 'destructive' | 'muted' | 'success' | 'warn' {
  if (fit === 'fits-gpu') {
    return 'success'
  }

  if (fit === 'needs-ram') {
    return 'warn'
  }

  if (fit === 'too-big') {
    return 'destructive'
  }

  return 'muted'
}

function browsedModelId(group: HFFileGroup): string {
  // Mirrors the backend's derivation: first file's name, split-part
  // suffix stripped — the id the download job carries.
  const first = group.paths[0].split('/').pop() ?? group.paths[0]

  return first.replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '')
}

function BrowseSection({ onChanged }: { onChanged: () => void }) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const jobs = useStore($localRuntimeJobs)
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<HFSearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [openRepo, setOpenRepo] = useState<null | string>(null)
  const [files, setFiles] = useState<HFFileGroup[]>([])
  const [listing, setListing] = useState(false)
  const [error, setError] = useState<null | string>(null)
  // Guard against the past: a stale search result must never overwrite a
  // newer query's hits (the desktop guide's out-of-order rule).
  const searchSeq = useRef(0)

  useEffect(() => {
    const q = query.trim()

    if (q.length < 2) {
      setHits([])
      setSearching(false)

      return
    }

    const seq = ++searchSeq.current
    setSearching(true)

    const handle = setTimeout(() => {
      searchHFModels(q)
        .then(r => {
          if (searchSeq.current === seq) {
            setHits(r.hits)
            setError(null)
          }
        })
        .catch((e: Error) => {
          if (searchSeq.current === seq) {
            setError(e.message)
          }
        })
        .finally(() => {
          if (searchSeq.current === seq) {
            setSearching(false)
          }
        })
    }, 350)

    return () => clearTimeout(handle)
  }, [query])

  const openFiles = useCallback((repo: string) => {
    setOpenRepo(repo)
    setFiles([])
    setListing(true)
    listHFRepoFiles(repo)
      .then(r => setFiles(r.files))
      .catch((e: Error) => setError(e.message))
      .finally(() => setListing(false))
  }, [])

  const startBrowsedDownload = useCallback(
    (repo: string, group: HFFileGroup) => {
      downloadBrowsedModel(repo, group.paths)
        .then(r => {
          if (r.already_downloaded) {
            notify({ durationMs: 3_000, kind: 'info', message: copy.browseAlreadyDownloaded, title: copy.browseTitle })

            return
          }

          // Same feedback loop as catalog downloads: the job store polls
          // and the tile renders live progress from it.
          watchLocalRuntimeJobs()
          notify({
            durationMs: 3_000,
            kind: 'info',
            message: copy.browseDownloadStarted.replace('{name}', r.model_id),
            title: copy.browseTitle
          })
          onChanged()
        })
        .catch((e: Error) => notifyError(e, copy.browseTitle))
    },
    [copy.browseAlreadyDownloaded, copy.browseDownloadStarted, copy.browseTitle, onChanged]
  )

  const sideload = useCallback(() => {
    window.hermesDesktop
      .selectPaths({ filters: [{ extensions: ['gguf'], name: 'GGUF models' }], title: copy.sideloadTitle })
      .then(paths => {
        if (!paths.length) {
          return
        }

        return sideloadLocalModel(paths[0]).then(r => {
          notify({
            durationMs: 3_000,
            kind: 'success',
            message: r.already_present ? copy.sideloadAlreadyPresent : copy.sideloadDone.replace('{name}', r.model_id),
            title: copy.browseTitle
          })
          onChanged()
        })
      })
      .catch((e: Error) => notifyError(e, copy.browseTitle))
  }, [copy.browseTitle, copy.sideloadAlreadyPresent, copy.sideloadDone, copy.sideloadTitle, onChanged])

  return (
    <SettingsSection
      aside={
        <Button onClick={sideload} size="sm" variant="outline">
          <FolderOpen className="mr-1 size-3.5" />
          {copy.sideloadButton}
        </Button>
      }
      icon={Search}
      title={copy.browseTitle}
    >
      <div id="local-model-browse">
        <p className="text-[0.75rem] text-muted-foreground">{copy.browseHint}</p>

        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            className="w-full rounded-md border border-(--ui-border) bg-transparent py-1.5 pl-8 pr-3 text-[0.8rem] outline-none placeholder:text-muted-foreground focus:border-primary"
            onChange={e => setQuery(e.target.value)}
            placeholder={copy.browsePlaceholder}
            value={query}
          />
        </div>

        {searching && (
          <p className="flex items-center gap-2 text-[0.75rem] text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            {copy.browseSearching}
          </p>
        )}

        {error && <p className="text-[0.75rem] text-destructive">{error}</p>}

        <div className="grid gap-1">
          {hits.map(hit => (
            <div key={hit.repo}>
              <ListRow
                action={
                  <Button onClick={() => openFiles(hit.repo)} size="sm" variant="ghost">
                    {openRepo === hit.repo ? copy.browseRefresh : copy.browseShowFiles}
                  </Button>
                }
                description={
                  <span>
                    {Intl.NumberFormat().format(hit.downloads)} {copy.browseDownloads}
                    {' · '}
                    {Intl.NumberFormat().format(hit.likes)} {copy.browseLikes}
                    {hit.gated ? ` · ${copy.browseGated}` : ''}
                  </span>
                }
                title={<span className="font-mono text-[0.8rem]">{hit.repo}</span>}
              />

              {openRepo === hit.repo && (
                <div className="ml-4 grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-1.5 border-l border-(--ui-border) py-1 pl-3">
                  {listing && (
                    <p className="col-span-full flex items-center gap-2 py-1 text-[0.75rem] text-muted-foreground">
                      <Loader2 className="size-3 animate-spin" />
                      {copy.browseListing}
                    </p>
                  )}

                  {!listing && files.length === 0 && (
                    <p className="col-span-full py-1 text-[0.75rem] text-muted-foreground">{copy.browseNoGguf}</p>
                  )}

                  {files.map(group => {
                    const dJob = runningDownloadFor(jobs, browsedModelId(group))

                    return (
                      <div
                        className="flex flex-col gap-1 rounded-md border border-(--ui-border) px-2.5 py-1.5"
                        key={group.label}
                      >
                        <span className="flex w-full items-center justify-between gap-2">
                          <span className="truncate font-mono text-[0.75rem]">
                            {group.label}
                            {group.paths.length > 1 ? ` ×${group.paths.length}` : ''}
                          </span>

                          <Button
                            aria-label={copy.browseDownloadAria.replace('{name}', group.label)}
                            className="h-6 shrink-0 px-2"
                            disabled={Boolean(dJob)}
                            onClick={() => startBrowsedDownload(hit.repo, group)}
                            size="sm"
                            variant="ghost"
                          >
                            {dJob ? <Loader2 className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}
                          </Button>
                        </span>

                        {dJob ? (
                          <>
                            <ProgressBar percent={dJob.percent} />

                            <span className="text-[0.68rem] text-muted-foreground">
                              {!dJob.done_bytes && dJob.detail
                                ? dJob.detail
                                : copy.downloadProgress(gbLabel(dJob.done_bytes), gbLabel(dJob.total_bytes))}
                            </span>
                          </>
                        ) : (
                          <span className="flex items-center justify-between gap-2">
                            <Pill tone={fitTone(group.fit)}>
                              <Cpu className="mr-1 size-3" />
                              {group.fit === 'fits-gpu'
                                ? copy.pillFitsGpu
                                : group.fit === 'needs-ram'
                                  ? copy.pillUsesRam
                                  : group.fit === 'too-big'
                                    ? copy.pillTooBig
                                    : copy.browseFitUnknown}
                            </Pill>

                            <span className="shrink-0 text-[0.7rem] text-muted-foreground">
                              {gbLabel(group.total_bytes)}
                            </span>
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </SettingsSection>
  )
}
