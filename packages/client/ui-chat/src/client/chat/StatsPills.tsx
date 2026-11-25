// Composer statistics pills, each its own 'conversation.composer.dock' list
// entry (activity, usage) so a plugin can replace or add one pill by id.
// Compact keeps speed and cache hit as plain readings; Detailed adds counts,
// token totals, and click-open dialogs. Every figure rides a durable session
// projection, never the paged window, so neither paging nor compaction can
// rescope a reading and no pill subscribes to the loaded nodes.

import { memo, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { IconDatabaseOutlineRegular, IconGaugeOutlineRegular, Tooltip } from '@deepseek-ai/dsh-client-ui-primitives'
import type { UseProjection } from '@deepseek-ai/dsh-api-session-controller/client'
import type { InjectFace, SnapshotSelectorHook } from '@deepseek-ai/dsh-client-ui-slots'
// Type-only: merges the sessionStats key into SessionProjectionMap for useProjection.
import type {} from '@deepseek-ai/dsh-session-stats/client'
import type { TokenUsageProjection } from '@deepseek-ai/dsh-token-meter/client'
import type { ChatViewSlotProps, PerformanceUsageInjected } from '../contract/slots.ts'
import type { ChatSnapshot } from '../contract/snapshot.ts'
import { formatTokensPerSecond } from './message-chrome.ts'
import { assistantStepReading } from '../contract/turn-metrics.ts'
import { formatCacheHitPercent, formatExactTokens, formatTokens } from './token-format.ts'
import { useStatDialog } from './stat-dialog.ts'
import css from './StatsPills.module.css'
import dialogCss from './stat-dialog.module.css'

/** Decode-timed steps the recent window keeps; mirrors RECENT_DECODE_WINDOW in the projection. */
const RECENT_DECODE_WINDOW = 10

interface WindowStats {
  turns: number
  steps: number
  /** Summed request wall time (step/start → assistant/message); 0 when no node carries timing. */
  llmMs: number
  /** Summed tool wall time (tool/call → tool/result); 0 when no pair is in-window. */
  toolMs: number
  /** Summed first-token latency over `ttftSteps`; 0 when no step records it. */
  ttftMs: number
  /** Steps carrying a recorded TTFT. */
  ttftSteps: number
  /** Summed decode wall time over steps that also report output tokens. */
  decodeMs: number
  /** Summed output tokens over the same decode-timed steps. */
  decodeTokens: number
  /** Steps in the most recent uninterrupted decode-timed run. */
  recentSteps: number
  /** Decode wall time over that run. */
  recentDecodeMs: number
  /** Output tokens over that run. */
  recentDecodeTokens: number
}

/**
 * Fold assistant and tool-result nodes into window-scoped totals — the
 * reference semantics for the projected figures, NOT a pill read path.
 *
 * It remains exported for the unit tests that pin each projected field's
 * definition (turns counted per distinct turn, ttft per timed step, decode
 * only over usage-carrying steps, tool time paired by call time), but no
 * component calls it: {@link useSessionStats} reads the `sessionStats`
 * projection alone. Folding a PAGED window yields the wrong answer for these
 * whole-log figures and costs an O(n) scan of every loaded node, so it must
 * not come back as a fallback.
 * @param nodes - snapshot nodes.
 * @returns window-scoped counts and summed wall times.
 */
export function deriveStats(nodes: ChatSnapshot['legacy']['nodes']): WindowStats {
  const turns = new Set<number>()
  let steps = 0
  let llmMs = 0
  let toolMs = 0
  let ttftMs = 0
  let ttftSteps = 0
  let decodeMs = 0
  let decodeTokens = 0
  const decodings: { ms: number; tokens: number }[] = []
  for (const node of nodes) {
    if (node.kind === 'tool-result') {
      if (node.callTime !== null) toolMs += Math.max(0, node.time - node.callTime)
      continue
    }
    if (node.kind !== 'assistant') continue
    turns.add(node.turn)
    steps += 1
    if (node.timing !== undefined && node.timing.stepStartTime !== null) {
      llmMs += Math.max(0, node.timing.completedTime - node.timing.stepStartTime)
    }
    const reading = assistantStepReading(node)
    if (reading.ttftMs !== null) {
      ttftMs += reading.ttftMs
      ttftSteps += 1
    }
    // One entry per decode-timed step, in order: the window keeps the last
    // RECENT_DECODE_WINDOW of them, and steps without decode timing are skipped
    // rather than breaking the run (mirroring the projection's window).
    const stepDecodeMs = reading.decodeMs
    const stepTokens = reading.outputTokens
    if (stepDecodeMs !== null && stepTokens !== null) {
      decodeMs += stepDecodeMs
      decodeTokens += stepTokens
      decodings.push({ ms: stepDecodeMs, tokens: stepTokens })
    }
  }
  const window = decodings.slice(-RECENT_DECODE_WINDOW)
  let recentSteps = 0
  let recentDecodeMs = 0
  let recentDecodeTokens = 0
  for (const entry of window) {
    recentSteps += 1
    recentDecodeMs += entry.ms
    recentDecodeTokens += entry.tokens
  }
  return {
    turns: turns.size, steps, llmMs, toolMs, ttftMs, ttftSteps, decodeMs, decodeTokens,
    recentSteps, recentDecodeMs, recentDecodeTokens,
  }
}

/**
 * Compact duration: 45.2s under a minute, 2m42s from there on.
 * @param ms - duration in milliseconds.
 * @returns display string.
 */
export function formatDuration(ms: number, t: ChatViewSlotProps['t']): string {
  const s = ms / 1_000
  if (s < 60) return t('duration.compactSeconds', { seconds: Math.round(s * 10) / 10 })
  const whole = Math.round(s)
  return t('duration.compactMinutes', {
    minutes: Math.floor(whole / 60),
    seconds: whole % 60,
  })
}

/**
 * Display-ready cache-hit share of prompt-side input over the whole durable log.
 * @param usage - the session's token-usage projection value.
 * @returns integer text when integer rounding stays below 100, otherwise the
 * minimum decimal precision that still rounds below 100; a full hit returns
 * 100, and no billed input returns null.
 */
export function cacheHitPercent(usage: TokenUsageProjection): string | null {
  const denominator = billedInputTokens(usage)
  return formatCacheHitPercent(usage.cacheReadTokens, denominator)
}

/**
 * Sum the three disjoint prompt-side billing buckets.
 * @param usage - the session's token-usage projection value.
 * @returns billed input tokens.
 */
export function billedInputTokens(usage: TokenUsageProjection): number {
  return usage.uncachedInputTokens + usage.cacheReadTokens + usage.cacheWriteTokens
}

type Translate = ChatViewSlotProps['t']

/** Props shared by every composer stats pill: the conversation-snapshot selector plus the projection read seat. */
export interface StatPillProps extends InjectFace<PerformanceUsageInjected> {
  useChat: SnapshotSelectorHook<ChatSnapshot>
  useProjection: UseProjection
  /** The owning dock's locale seat. */
  t: Translate
}

/** Dock entry id carried as `data-composer-stat` on each pill. */
type StatId = 'activity' | 'usage'

interface PillContent {
  stat: StatId
  icon: ReactNode
  label: ReactNode
  /** The full reading as plain text: the accessible name and the tooltip shown while the label is cut off. */
  text: string
}

function exactCount(value: number, t: Translate): string {
  return t('message.turnUsage.count', { count: formatExactTokens(value, t) })
}

/** One or two reading segments, visually joined by a separator glyph. */
function reading(first: string, second: string | null): Pick<PillContent, 'label' | 'text'> {
  if (second === null) return { label: first, text: first }
  return {
    label: (
      <>
        {first}
        <span className={css.sep} aria-hidden>·</span>
        {second}
      </>
    ),
    text: `${first} · ${second}`,
  }
}

/**
 * Every figure rides the durable `sessionStats` projection, so paging and
 * compaction cannot change any of them, and this pill subscribes to no part of
 * the paged window: without the projection value there is nothing durable to
 * show, so the pill renders nothing rather than folding the loaded window.
 *
 * The window fold (`deriveStats`) is deliberately NOT a read path. It walked
 * every settled node on each recomputation, which put an O(n) scan of the whole
 * loaded window on the composer dock of long sessions. Its per-field semantics
 * stay pinned by the unit tests, and folding a paged window is the wrong answer
 * for these figures anyway: the counts, time split, and throughput are
 * whole-log facts that a partial page cannot reconstruct.
 *
 * `dsh-session-stats` is a declared dependency of this package and is composed
 * by every official assembly, so the absent-value branch is the cold-start
 * window before the unit serves a value, not a supported operating mode.
 *
 * @param useProjection - projection read seat; the only data source.
 * @returns the whole-log figures, or `null` while the projection serves none.
 */
function useSessionStats(useProjection: UseProjection): WindowStats | null {
  return useProjection('sessionStats') ?? null
}

/**
 * The speed reading. Recent-run throughput answers "how fast now" and is shown
 * first because the whole-log ratio drifts toward older, denser steps and can
 * never climb back; the whole-log average follows whenever the two differ, so
 * neither figure is silently the other. A run of 0 means no decode-timed step
 * closed yet, in which case only the whole-log figure (or nothing) remains.
 *
 * @param stats - whole-log figures plus the recent run.
 * @param t - translate seat supplying the speed reading.
 * @returns the reading text, or null when no figure is available.
 */
function decodeSpeed(stats: WindowStats, t: Translate): string | null {
  const recent = stats.recentSteps > 0 && stats.recentDecodeMs > 0
    ? formatTokensPerSecond(stats.recentDecodeTokens / (stats.recentDecodeMs / 1_000))
    : null
  const whole = stats.decodeMs > 0
    ? formatTokensPerSecond(stats.decodeTokens / (stats.decodeMs / 1_000))
    : null
  if (recent === null) return whole === null ? null : t('message.tokensPerSecond', { tps: whole })
  if (whole === null || recent === whole) return t('message.tokensPerSecond', { tps: recent })
  return t('message.tokensPerSecondRecent', { recent, whole })
}

/** The full reading above a pill, only while an ellipsis cuts its label; the delay matches the dock's context meter. */
const FULL_READING_TOOLTIP = { side: 'top', portal: true, delayMs: 200, onlyWhenClipped: true } as const

/** A static reading: used when the pill has no dialog rows or the mode is Compact. */
function PlainPill({ stat, icon, label, text }: PillContent) {
  return (
    <span className={css.anchor} data-composer-stat={stat}>
      {/* No dialog carries a static reading, so a tap also pins the full text. */}
      <Tooltip label={text} {...FULL_READING_TOOLTIP} openOnClick>
        <span className={css.pill}>
          {icon}
          <span className={css.label}>{label}</span>
        </span>
      </Tooltip>
    </span>
  )
}

/**
 * A pill button opening its own portaled dialog. Each pill owns its open
 * state; useStatDialog closes it on Escape or an outside pointerdown or click,
 * so at most one dialog is open across the dock.
 */
function DialogPill({ stat, icon, label, text, title, titleValue, children }: PillContent & {
  title: string
  titleValue?: string
  children: ReactNode
}) {
  const { open, setOpen, rootRef, panelRef, anchorName } = useStatDialog()
  return (
    <span ref={rootRef} style={{ anchorName }} className={css.anchor} data-composer-stat={stat}>
      {/* The open dialog already shows every figure; a tooltip would cover it. */}
      <Tooltip label={text} {...FULL_READING_TOOLTIP} disabled={open}>
        <button
          type="button"
          className={css.pill}
          aria-haspopup="dialog"
          aria-expanded={open}
          aria-label={text}
          onClick={() => { setOpen(!open) }}
        >
          {icon}
          <span className={css.label}>{label}</span>
        </button>
      </Tooltip>
      {open && createPortal(
        <div
          ref={panelRef}
          className={dialogCss.panel}
          role="dialog"
          aria-label={title}
          style={{ positionAnchor: anchorName }}
        >
          <div className={dialogCss.title}>
            <span className={dialogCss.titleLabel}>
              {icon}
              {title}
            </span>
            {titleValue !== undefined && <span className={dialogCss.titleValue}>{titleValue}</span>}
          </div>
          <div className={dialogCss.titleRule} aria-hidden />
          {children}
        </div>,
        document.body,
      )}
    </span>
  )
}

/**
 * Turn and step counts with whole-session speed, opening the time and speed
 * dialog; Compact keeps only the speed reading. Renders nothing until the
 * `sessionStats` projection serves a value, since the paged window is not an
 * accounting source (see {@link useSessionStats}).
 */
export const ActivityPill = memo(function ActivityPill({ useProjection, usePerformanceUsage, t }: StatPillProps) {
  const mode = usePerformanceUsage(value => value)
  const stats = useSessionStats(useProjection)
  if (stats === null) return null
  const speed = decodeSpeed(stats, t)
  const icon = <IconGaugeOutlineRegular />
  if (mode === 'compact') return speed === null ? null : <PlainPill stat="activity" icon={icon} {...reading(speed, null)} />
  if (stats.steps === 0) return null
  const counts = t('stats.counts', { turns: stats.turns, steps: stats.steps })
  const content: PillContent = { stat: 'activity', icon, ...reading(counts, speed) }
  // A window without one timed figure has no dialog rows to show, so the pill
  // stays a plain reading instead of a button opening an empty dialog.
  if (stats.llmMs <= 0 && stats.toolMs <= 0 && stats.ttftSteps <= 0 && speed === null) return <PlainPill {...content} />
  return (
    // The counts head the dialog, so a cut-off reading stays complete once the dialog opens.
    <DialogPill {...content} title={t('stats.dialog.title')} titleValue={counts}>
      <dl className={dialogCss.details} data-session-stats-details>
        {stats.llmMs > 0 && (
          <>
            <dt>{t('stats.dialog.llmTime')}</dt>
            <dd>{formatDuration(stats.llmMs, t)}</dd>
          </>
        )}
        {stats.toolMs > 0 && (
          <>
            <dt>{t('stats.dialog.toolTime')}</dt>
            <dd>{formatDuration(stats.toolMs, t)}</dd>
          </>
        )}
        {stats.ttftSteps > 0 && (
          <>
            <dt>{t('stats.dialog.ttft')}</dt>
            <dd>{formatDuration(stats.ttftMs / stats.ttftSteps, t)}</dd>
          </>
        )}
        {speed !== null && (
          <>
            <dt>{t('stats.dialog.speed')}</dt>
            <dd>{speed}</dd>
          </>
        )}
      </dl>
    </DialogPill>
  )
})

/** Whole-log token total and cache hit; Compact keeps only the cache hit. */
export const UsagePill = memo(function UsagePill({ useProjection, usePerformanceUsage, t }: StatPillProps) {
  const mode = usePerformanceUsage(value => value)
  const usage = useProjection('tokenUsage')
  // Gated on actual token activity: a session whose steps all settled without
  // billing (e.g. every request failed) shows no usage pill.
  if (usage === undefined || (billedInputTokens(usage) === 0 && usage.outputTokens === 0)) return null
  const cacheHit = cacheHitPercent(usage)
  const cacheHitText = cacheHit !== null ? t('stats.cacheHit', { percent: cacheHit }) : null
  const icon = <IconDatabaseOutlineRegular />
  if (mode === 'compact') {
    return cacheHitText === null ? null : <PlainPill stat="usage" icon={icon} {...reading(cacheHitText, null)} />
  }
  // Same aggregate as the Turn pill's totalTokens: every prompt-side billing bucket plus output.
  const total = billedInputTokens(usage) + usage.outputTokens
  const totalText = t('message.turnUsage.count', { count: formatTokens(total, t) })
  return (
    <DialogPill
      stat="usage"
      icon={icon}
      {...reading(totalText, cacheHitText)}
      title={t('stats.dialog.usageTitle')}
      titleValue={exactCount(total, t)}
    >
      {/* jscpd:ignore-start -- the session-total bucket rows deliberately mirror
          TurnUsagePanel's per-turn dl: same skin, different data contract (the
          buckets are always present here; per-turn fields are optional). A
          session that never wrote cache drops the row, as the per-turn panel
          drops its absent fields. */}
      <dl className={dialogCss.details} data-session-stats-usage>
        {cacheHit !== null && (
          <>
            <dt>{t('message.turnUsage.cacheHit')}</dt>
            <dd>{`${cacheHit}%`}</dd>
          </>
        )}
        <dt>{t('message.turnUsage.input')}</dt>
        <dd>{exactCount(usage.uncachedInputTokens, t)}</dd>
        <dt>{t('message.turnUsage.cacheRead')}</dt>
        <dd>{exactCount(usage.cacheReadTokens, t)}</dd>
        {usage.cacheWriteTokens !== 0 && (
          <>
            <dt>{t('message.turnUsage.cacheWrite')}</dt>
            <dd>{exactCount(usage.cacheWriteTokens, t)}</dd>
          </>
        )}
        <dt>{t('message.turnUsage.output')}</dt>
        <dd>{exactCount(usage.outputTokens, t)}</dd>
      </dl>
      {/* jscpd:ignore-end */}
    </DialogPill>
  )
})
