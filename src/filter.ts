import type { Part, Message } from "@opencode-ai/sdk"
import type { Embedder } from "./embedder.js"
import type { PluginOptions } from "./config.js"

const MAX_EMBED_CHARS = 1000
export const PRUNED_PLACEHOLDER = ""

// ---------------------------------------------------------------------------
// Internal types matching the opencode SDK ToolPart at runtime
// ---------------------------------------------------------------------------

interface ToolStateCompleted {
  status: "completed"
  output: string
  input: Record<string, unknown>
  title: string
}

interface ToolPart {
  type: "tool"
  tool: string
  state: { status: string } & Partial<ToolStateCompleted>
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom < 1e-10 ? 0 : dot / denom
}

export function matchesPattern(name: string, patterns: string[]): boolean {
  return patterns.some((p) => name === p || name.startsWith(p + "__") || name.startsWith(p + "_"))
}

function isCompletedToolPart(part: Part): part is Part & ToolPart {
  return part.type === "tool" && (part as unknown as ToolPart).state.status === "completed"
}

function embedText(part: ToolPart): string {
  const out = part.state.output ?? ""
  return `${part.tool}: ${out.slice(0, MAX_EMBED_CHARS)}`
}

// ---------------------------------------------------------------------------
// Extract latest user query from hook messages
// ---------------------------------------------------------------------------

export function extractLatestUserQuery(
  messages: Array<{ info: Message; parts: Part[] }>
): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].info.role !== "user") continue
    const text = messages[i].parts
      .filter((p) => p.type === "text")
      .map((p) => (p as { text: string }).text)
      .join(" ")
      .trim()
    if (text) return text
  }
  return ""
}

// ---------------------------------------------------------------------------
// Prune irrelevant completed tool outputs in-place
// ---------------------------------------------------------------------------

export async function pruneToolOutputs(
  messages: Array<{ info: Message; parts: Part[] }>,
  query: string,
  opts: PluginOptions,
  embedder: Embedder
): Promise<number> {
  const candidates: Array<{ part: ToolPart }> = []
  const pinned: Set<ToolPart> = new Set()

  for (const { parts } of messages) {
    for (const part of parts) {
      if (!isCompletedToolPart(part)) continue
      const tp = part as unknown as ToolPart
      if (matchesPattern(tp.tool, opts.alwaysInclude)) {
        pinned.add(tp)
      } else {
        candidates.push({ part: tp })
      }
    }
  }

  if (candidates.length === 0) return 0

  const queryEmb = await embedder.embed(query, "query")
  const scores = await Promise.all(
    candidates.map(async ({ part }) => {
      const emb = await embedder.embed(embedText(part), "document")
      return { part, score: cosineSimilarity(queryEmb, emb) }
    })
  )

  scores.sort((a, b) => b.score - a.score)
  const keep = new Set(
    scores
      .slice(0, opts.topK)
      .filter((s) => s.score >= opts.minSimilarity)
      .map((s) => s.part)
  )

  let pruned = 0
  for (const { part } of candidates) {
    if (!keep.has(part)) {
      part.state = { ...part.state, output: PRUNED_PLACEHOLDER }
      pruned++
    }
  }
  return pruned
}
