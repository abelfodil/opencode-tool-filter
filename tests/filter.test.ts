import { describe, it, expect } from "vitest"
import { extractLatestUserQuery, pruneToolOutputs, PRUNED_PLACEHOLDER } from "../src/filter.js"
import { PluginOptionsSchema } from "../src/config.js"
import type { Embedder } from "../src/embedder.js"
import type { Part, Message } from "@opencode-ai/sdk"

// ---------------------------------------------------------------------------
// Mock embedder — deterministic 2D embeddings based on keywords
//   [0] = file relevance  [1] = search relevance
// ---------------------------------------------------------------------------

function keywords(text: string): [number, number] {
  const t = text.toLowerCase()
  const file = t.includes("file") || t.includes("read") || t.includes("write") ? 1 : 0
  const search = t.includes("search") || t.includes("web") || t.includes("query") ? 1 : 0
  if (file === 0 && search === 0) return [0.707, 0.707]
  const mag = Math.sqrt(file ** 2 + search ** 2)
  return [file / mag, search / mag]
}

const mockEmbedder = {
  async embed(text: string): Promise<Float32Array> {
    return Float32Array.from(keywords(text))
  },
} as unknown as Embedder

// ---------------------------------------------------------------------------
// Helpers to build fake opencode message structures
// ---------------------------------------------------------------------------

function userMsg(text: string): { info: Message; parts: Part[] } {
  return {
    info: { role: "user" } as Message,
    parts: [{ type: "text", text } as Part],
  }
}

function assistantMsgWithTools(
  tools: Array<{ name: string; output: string }>
): { info: Message; parts: Part[] } {
  return {
    info: { role: "assistant" } as Message,
    parts: tools.map((t) => ({
      type: "tool",
      tool: t.name,
      callID: t.name + "-id",
      state: {
        status: "completed",
        output: t.output,
        input: {},
        title: t.name,
        metadata: {},
        time: { start: 0, end: 1 },
      },
    })) as unknown as Part[],
  }
}

const defaultOpts = PluginOptionsSchema.parse({})

// ---------------------------------------------------------------------------
// extractLatestUserQuery
// ---------------------------------------------------------------------------

describe("extractLatestUserQuery", () => {
  it("returns empty string for empty messages", () => {
    expect(extractLatestUserQuery([])).toBe("")
  })

  it("returns empty string when no user messages", () => {
    expect(extractLatestUserQuery([assistantMsgWithTools([])])).toBe("")
  })

  it("extracts text from a user message", () => {
    expect(extractLatestUserQuery([userMsg("search the web")])).toBe("search the web")
  })

  it("returns the LAST user message when multiple exist", () => {
    expect(
      extractLatestUserQuery([userMsg("first"), assistantMsgWithTools([]), userMsg("last")])
    ).toBe("last")
  })
})

// ---------------------------------------------------------------------------
// pruneToolOutputs
// ---------------------------------------------------------------------------

describe("pruneToolOutputs – basic", () => {
  it("returns 0 when there are no tool parts", async () => {
    const msgs = [userMsg("search")]
    expect(await pruneToolOutputs(msgs, "search", defaultOpts, mockEmbedder)).toBe(0)
  })

  it("keeps all tools when count ≤ topK", async () => {
    const msgs = [
      assistantMsgWithTools([
        { name: "readFile", output: "file contents" },
        { name: "webSearch", output: "search results" },
      ]),
      userMsg("search"),
    ]
    const opts = PluginOptionsSchema.parse({ topK: 10, minSimilarity: 0 })
    const pruned = await pruneToolOutputs(msgs, "search the web", opts, mockEmbedder)
    expect(pruned).toBe(0)
  })
})

describe("pruneToolOutputs – ranking", () => {
  it("prunes file tool for a search query when topK=1", async () => {
    const filePart = { name: "readFile", output: "file contents" }
    const searchPart = { name: "webSearch", output: "search results from web" }
    const msgs = [assistantMsgWithTools([filePart, searchPart]), userMsg("search")]
    const opts = PluginOptionsSchema.parse({ topK: 1, minSimilarity: 0 })

    await pruneToolOutputs(msgs, "search the web", opts, mockEmbedder)

    const parts = msgs[0].parts as any[]
    const readFilePart = parts.find((p) => p.tool === "readFile")
    const webSearchPart = parts.find((p) => p.tool === "webSearch")
    expect(readFilePart.state.output).toBe(PRUNED_PLACEHOLDER)
    expect(webSearchPart.state.output).toBe("search results from web")
  })

  it("prunes search tool for a file query when topK=1", async () => {
    const msgs = [
      assistantMsgWithTools([
        { name: "readFile", output: "file contents" },
        { name: "webSearch", output: "search results from web" },
      ]),
      userMsg("read"),
    ]
    const opts = PluginOptionsSchema.parse({ topK: 1, minSimilarity: 0 })

    await pruneToolOutputs(msgs, "read a file from disk", opts, mockEmbedder)

    const parts = msgs[0].parts as any[]
    expect(parts.find((p) => p.tool === "webSearch").state.output).toBe(PRUNED_PLACEHOLDER)
    expect(parts.find((p) => p.tool === "readFile").state.output).toBe("file contents")
  })
})

describe("pruneToolOutputs – alwaysInclude", () => {
  it("never prunes tools matching alwaysInclude", async () => {
    const msgs = [
      assistantMsgWithTools([
        { name: "readFile", output: "file contents" },
        { name: "webSearch", output: "search results" },
      ]),
      userMsg("search"),
    ]
    const opts = PluginOptionsSchema.parse({ topK: 1, minSimilarity: 0, alwaysInclude: ["readFile"] })

    await pruneToolOutputs(msgs, "search the web", opts, mockEmbedder)

    const parts = msgs[0].parts as any[]
    expect(parts.find((p) => p.tool === "readFile").state.output).toBe("file contents")
  })

  it("pins tools matching a prefix", async () => {
    const msgs = [
      assistantMsgWithTools([
        { name: "mcp__files__read", output: "mcp file contents" },
        { name: "webSearch", output: "search results" },
      ]),
      userMsg("search"),
    ]
    const opts = PluginOptionsSchema.parse({ topK: 1, minSimilarity: 0, alwaysInclude: ["mcp__files"] })

    await pruneToolOutputs(msgs, "search the web", opts, mockEmbedder)

    const parts = msgs[0].parts as any[]
    expect(parts.find((p) => p.tool === "mcp__files__read").state.output).toBe("mcp file contents")
  })
})

describe("pruneToolOutputs – minSimilarity", () => {
  it("prunes tools below the similarity threshold", async () => {
    // neutral tool embedds to [0.707, 0.707]; search query to [0, 1]
    // cosine ≈ 0.707, below threshold 0.9 → pruned
    const msgs = [
      assistantMsgWithTools([
        { name: "webSearch", output: "search results from web query" },
        { name: "ping", output: "pong" },
      ]),
      userMsg("search"),
    ]
    const opts = PluginOptionsSchema.parse({ topK: 10, minSimilarity: 0.9 })

    await pruneToolOutputs(msgs, "search the web", opts, mockEmbedder)

    const parts = msgs[0].parts as any[]
    expect(parts.find((p) => p.tool === "webSearch").state.output).toBe("search results from web query")
    expect(parts.find((p) => p.tool === "ping").state.output).toBe(PRUNED_PLACEHOLDER)
  })
})
