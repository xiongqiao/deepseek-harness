/**
 * The `sessionStats` projection unit: a pure fold of step boundaries, stream
 * embedded streams, tool pairs, and assembled assistant messages into whole-log counts
 * and wall times.
 *
 * `step/end` — not `assistant/message` — is the counted step event because it
 * is the step lifecycle authority: the loop appends exactly one per entered
 * step, in a `finally`, so completed, failed, cancelled, and max-tokens steps
 * all land one. Counting assembled assistant messages instead would overcount
 * max-tokens usage-host messages (empty content, excluded from the surface)
 * and undercount cancelled steps (aborted before the message assembles).
 *
 * The wall-time folds mirror the client window fold field by field
 * (`deriveStats` in dsh-client-ui-conversation, that fold's whole-window
 * fallback role): model time is `step/start` → `assistant/message`, first
 * token is the first non-empty delta chunk and survives an in-step
 * `llm/retry`, decode spans first token → assembled message on steps that
 * also report output tokens, and tool time pairs `tool/call` → `tool/result`
 * by callId. A cancelled step assembles no message, so its partial stream
 * time stays uncounted in every time figure — matching the window, which
 * renders it as an untimed interrupted node.
 *
 * @module @deepseek-ai/dsh-session-stats/projection
 */

import { z } from 'zod'
import { assistantStreamFirstTokenTime } from '@deepseek-ai/dsh-llm'
import type { ProjectionDefinition } from '@deepseek-ai/dsh-session-projection'


/** Accumulated whole-log figures (the view is exactly these totals). */
interface SessionStatsTotals {
  /** Distinct turns with at least one closed step so far. */
  turns: number
  /** Closed steps so far. */
  steps: number
  /** Summed model wall time over message-assembling steps, ms. */
  llmMs: number
  /** Summed matched tool call→result wall time, ms. */
  toolMs: number
  /** Summed first-token latency over `ttftSteps`, ms. */
  ttftMs: number
  /** Steps carrying a recorded first token. */
  ttftSteps: number
  /** Summed decode wall time over usage-reporting steps, ms. */
  decodeMs: number
  /** Summed provider output tokens over the same steps. */
  decodeTokens: number
}

/**
 * Steps retained by the recent-decode window. Ten is long enough that one short
 * or long step cannot swing the reading, and short enough that the window
 * follows a real speed change within a screen of messages.
 */
const RECENT_DECODE_WINDOW = 10

/** One decode-timed step's contribution to the recent window. */
interface DecodeSample {
  /** Decode wall time for the step, ms. */
  ms: number
  /** Provider output tokens for the same step. */
  tokens: number
}

/**
 * Recent-decode window: the last {@link RECENT_DECODE_WINDOW} decode-timed
 * steps, oldest first.
 *
 * It answers "how fast is decoding right now", which the monotonic whole-log
 * average in {@link SessionStatsTotals} cannot: that ratio keeps every step ever
 * taken, so a session that starts with long answers and continues with short
 * ones drifts down and never recovers, even when the model is as fast as ever.
 *
 * The window is bounded, so it carries history and cannot be an optional
 * addition — which is exactly why `stateVersion` is 2 and stored rows refold
 * once. (An earlier "consecutive run" design avoided the bump by depending only
 * on increments, but it restarts solely at a step that reported no decode
 * timing; in a session whose steps all report timing it equals the whole-log
 * average and would fix nothing.)
 */
type SessionStatsRecent = DecodeSample[]

/**
 * Fold state: the totals plus the in-flight boundaries they accrue from.
 * Turn numbers are host-assigned and monotonic per session, so a single
 * `lastTurn` slot decides "first closed step of a new turn"; the state is
 * plain JSON per the unit contract (persisted-cache precondition).
 */
interface SessionStatsState extends SessionStatsTotals {
  /** Turn of the last counted `step/end`; null before the first. */
  lastTurn: number | null
  /** The open step's boundary facts; null outside a step or after its message assembled. */
  openStep: { turn: number; step: number; startTime: number; firstTokenTime: number | null } | null
  /** Dispatch times of tool calls whose result has not landed, by callId. */
  pendingCalls: Record<string, number>
  /**
   * Recent-decode window, oldest first. Required (not optional) because
   * `stateVersion` 2 already invalidates every row written without it, and an
   * explicit array avoids Zod's optional output, which
   * `exactOptionalPropertyTypes` rejects for a `recent?:` field.
   */
  recent: SessionStatsRecent
  /**
   * The recently closed step whose decode the window has not folded yet; it is
   * created by `assistant/message` and consumed by the following `step/end`,
   * because decode timing belongs to the message while the step is the counted
   * boundary. `null` while no such step is pending.
   */
  recentPending: { decodeMs: number; decodeTokens: number | null } | null
}

const sessionStatsSchema = z.object({
  turns: z.number().int().nonnegative(),
  steps: z.number().int().nonnegative(),
  llmMs: z.number().nonnegative(),
  toolMs: z.number().nonnegative(),
  ttftMs: z.number().nonnegative(),
  ttftSteps: z.number().int().nonnegative(),
  decodeMs: z.number().nonnegative(),
  decodeTokens: z.number().nonnegative(),
}).strict()

/**
 * The fold state's shape (totals plus in-flight boundaries), validated on
 * persisted-cache rows after their `ver` gate — the unit's input boundary.
 * The view is a strict subset of the state, so this schema extends
 * `sessionStatsSchema` (the wire output boundary) with the boundary fields.
 */
const sessionStatsStateSchema = sessionStatsSchema.extend({
  lastTurn: z.number().int().nonnegative().nullable(),
  openStep: z.object({
    turn: z.number().int().nonnegative(),
    step: z.number().int().nonnegative(),
    startTime: z.number().nonnegative(),
    firstTokenTime: z.number().nonnegative().nullable(),
  }).nullable(),
  pendingCalls: z.record(z.string(), z.number().nonnegative()),
  // Required: `stateVersion` 2 invalidates every row written without them, so a
  // v1 row fails this schema and refolds instead of being read as a short window.
  recent: z.array(z.object({
    ms: z.number().nonnegative(),
    tokens: z.number().nonnegative(),
  })).max(RECENT_DECODE_WINDOW),
  recentPending: z.object({
    decodeMs: z.number().nonnegative(),
    decodeTokens: z.number().nonnegative().nullable(),
  }).nullable(),
})

/**
 * Provider-reported completion tokens, guarded the way the window fold guards
 * node usage.
 * @param usage - the assistant/message event's optional usage record.
 * @returns the output-token count, or null when unreported or invalid.
 */
function usageOutputTokens(usage: unknown): number | null {
  if (typeof usage !== 'object' || usage === null) return null
  const value = (usage as { outputTokens?: unknown }).outputTokens
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

/** The `sessionStats` unit registered on `ctx.sessionProjections` (exported for the unit spec). */
export const sessionStatsProjectionDefinition = {
  key: 'sessionStats',
  /**
   * Bumped to 2 by the recent-decode window. A bounded window carries history,
   * so it cannot be an optional addition: a row written before the field existed
   * has no window to continue from, and replaying a tail over it cannot equal a
   * full fold. The contract requires a bump in exactly that case, so stored rows
   * refold once.
   */
  stateVersion: 2,
  stateSchema: sessionStatsStateSchema,
  init: () => ({
    turns: 0,
    steps: 0,
    llmMs: 0,
    toolMs: 0,
    ttftMs: 0,
    ttftSteps: 0,
    decodeMs: 0,
    decodeTokens: 0,
    lastTurn: null,
    openStep: null,
    pendingCalls: {},
    recent: [],
    recentPending: null,
  }),
  apply: (state, event) => {
    // Every uninteresting event returns the same reference (Object.is gates the change feed).
    switch (event.type) {
      case 'step/start':
        return {
          ...state,
          openStep: { turn: event.data.turn, step: event.data.step, startTime: event.time, firstTokenTime: null },
        }
      case 'assistant/attempt': {
        const open = state.openStep
        if (open === null || open.turn !== event.data.turn || open.step !== event.data.step) return state
        const first = assistantStreamFirstTokenTime(event.data.stream) ?? null
        if (open.firstTokenTime !== null || first === null) return state
        return { ...state, openStep: { ...open, firstTokenTime: first } }
      }
      case 'assistant/message': {
        const open = state.openStep
        if (open === null || open.turn !== event.data.turn || open.step !== event.data.step) return state
        const firstToken = open.firstTokenTime ?? assistantStreamFirstTokenTime(event.data.stream) ?? null
        // One assembled message per step: closing the boundary means a
        // defensive duplicate cannot accrue twice.
        const next: SessionStatsState = {
          ...state,
          llmMs: state.llmMs + Math.max(0, event.time - open.startTime),
          openStep: null,
        }
        if (firstToken !== null) {
          next.ttftMs += Math.max(0, firstToken - open.startTime)
          next.ttftSteps += 1
          const outputTokens = usageOutputTokens(event.data.usage)
          if (outputTokens !== null) {
            next.decodeMs += Math.max(0, event.time - firstToken)
            next.decodeTokens += outputTokens
          }
          // Hand the decode reading to the step/end that closes this step: the
          // run counts step boundaries, so only there does a step join it.
          next.recentPending = { decodeMs: Math.max(0, event.time - firstToken), decodeTokens: outputTokens }
        } else {
          next.recentPending = null
        }
        return next
      }
      case 'tool/call':
        return { ...state, pendingCalls: { ...state.pendingCalls, [event.data.callId]: event.time } }
      case 'tool/result': {
        // Own-key check: callId is provider-minted (model/tool JSON boundary),
        // so a prototype property name ('constructor', 'toString') on a result
        // with no recorded call must read as unmatched, not as an inherited
        // function that would poison toolMs with NaN.
        const callId = event.data.message.source.callId
        const dispatched = Object.hasOwn(state.pendingCalls, callId) ? state.pendingCalls[callId] : undefined
        if (dispatched === undefined) return state
        const pendingCalls = Object.fromEntries(
          Object.entries(state.pendingCalls).filter(([id]) => id !== callId),
        )
        return { ...state, toolMs: state.toolMs + Math.max(0, event.time - dispatched), pendingCalls }
      }
      case 'step/end': {
        const turns = state.lastTurn === event.data.turn ? state.turns : state.turns + 1
        // A step that reported decode timing joins the window; one that did not
        // (cancelled before a message assembled, or no usage) contributes
        // nothing and leaves the window intact, so tool-only steps do not blank
        // the reading.
        const pending = state.recentPending
        const recent: SessionStatsRecent = pending !== null && pending.decodeTokens !== null && pending.decodeMs > 0
          ? [...state.recent, { ms: pending.decodeMs, tokens: pending.decodeTokens }].slice(-RECENT_DECODE_WINDOW)
          : state.recent
        return {
          ...state,
          turns,
          steps: state.steps + 1,
          lastTurn: event.data.turn,
          openStep: null,
          recent,
          recentPending: null,
        }
      }
      case 'turn/end':
        // A call whose result never landed belongs to a cancelled or failed
        // turn; results always land within their turn, so drop the leftovers
        // instead of growing persisted state forever.
        return Object.keys(state.pendingCalls).length === 0 ? state : { ...state, pendingCalls: {} }
      default:
        return state
    }
  },
  wire: {
    viewSchema: sessionStatsSchema.extend({
      recentSteps: z.number().int().nonnegative(),
      recentDecodeMs: z.number().nonnegative(),
      recentDecodeTokens: z.number().nonnegative(),
    }).strict(),
    view: (state) => {
      let recentSteps = 0
      let recentDecodeMs = 0
      let recentDecodeTokens = 0
      for (const sample of state.recent) {
        recentSteps += 1
        recentDecodeMs += sample.ms
        recentDecodeTokens += sample.tokens
      }
      return {
        turns: state.turns,
        steps: state.steps,
        llmMs: state.llmMs,
        toolMs: state.toolMs,
        ttftMs: state.ttftMs,
        ttftSteps: state.ttftSteps,
        decodeMs: state.decodeMs,
        decodeTokens: state.decodeTokens,
        recentSteps,
        recentDecodeMs,
        recentDecodeTokens,
      }
    },
    summary: false,
  },
} satisfies ProjectionDefinition<'sessionStats', SessionStatsState>
