import type { Plugin } from "@opencode-ai/plugin"
import { PluginOptionsSchema } from "./config.js"
import { Embedder } from "./embedder.js"
import { cosineSimilarity, matchesPattern } from "./filter.js"

const plugin: Plugin = async (_input, rawOptions = {}) => {
  const opts = PluginOptionsSchema.parse(rawOptions)
  if (!opts.enabled) return {}

  const embedder = new Embedder({ model: opts.model, dtype: opts.dtype })
  embedder.warmup().catch((err) =>
    console.warn("[tool-filter] Embedding warm-up failed:", err)
  )

  // Persists across requests: toolID → description
  // Populated by tool.definition; used in chat.message to pre-exclude tools
  const toolCache = new Map<string, string>()

  let lastQuery = ""

  return {
    // ------------------------------------------------------------------
    // Fires when the user sends a message — before resolveTools runs.
    // On the second+ request (once toolCache is warm), we score every
    // known tool and set output.message.tools[id] = false for irrelevant
    // ones. resolveTools() filters those out entirely — true removal,
    // no blanking, no risk of tool-call errors.
    // ------------------------------------------------------------------
    "chat.message": async (_input, output) => {
      const text = (output.parts as any[])
        .filter((p) => p.type === "text")
        .map((p) => p.text as string)
        .join(" ")
        .trim()
      if (text) lastQuery = text

      if (toolCache.size === 0 || !lastQuery) return

      const queryEmb = await embedder.embed(lastQuery, "query")
      let excluded = 0

      for (const [toolID, desc] of toolCache) {
        if (matchesPattern(toolID, opts.alwaysInclude)) continue

        const toolText = desc ? `${toolID}: ${desc}` : toolID
        const toolEmb = await embedder.embed(toolText, "document")
        const score = cosineSimilarity(queryEmb, toolEmb)

        if (score < opts.minSimilarity) {
          const msg = output.message as any
          msg.tools ??= {}
          msg.tools[toolID] = false
          excluded++
        }
      }

      if (excluded > 0) {
        console.log(`[tool-filter] excluded ${excluded}/${toolCache.size} tools from request`)
      }
    },

    // ------------------------------------------------------------------
    // Fires per tool during request preparation.
    // Updates the cache so chat.message can use it next request.
    // On the very first request (cache cold), also blanks descriptions
    // as a fallback — better than sending full definitions untouched.
    // ------------------------------------------------------------------
    "tool.definition": async ({ toolID }, output) => {
      // Always update cache with latest description
      toolCache.set(toolID, output.description)

      // Fallback for first request: blank description for irrelevant tools
      // (tool is still sent, but LLM is less likely to call it without context)
      if (!lastQuery || toolCache.size <= 1) return
      if (matchesPattern(toolID, opts.alwaysInclude)) return

      const queryEmb = await embedder.embed(lastQuery, "query")
      const toolText = output.description ? `${toolID}: ${output.description}` : toolID
      const toolEmb = await embedder.embed(toolText, "document")
      const score = cosineSimilarity(queryEmb, toolEmb)

      if (score < opts.minSimilarity) {
        output.description = ""
      }
    },
  }
}

export default plugin
