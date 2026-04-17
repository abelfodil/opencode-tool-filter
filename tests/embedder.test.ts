import { describe, it, expect, vi, beforeEach } from "vitest"

// ---------------------------------------------------------------------------
// Mock @huggingface/transformers before importing Embedder
// ---------------------------------------------------------------------------

const mockPipelineFn = vi.fn(async (text: string) => ({
  data: new Float32Array([0.1, 0.2, 0.3]),
}))

const mockPipelineFactory = vi.fn(async () => mockPipelineFn)

vi.mock("@huggingface/transformers", () => ({
  pipeline: mockPipelineFactory,
}))

const { Embedder } = await import("../src/embedder.js")

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks()
})

describe("Embedder – model loading", () => {
  it("does not load the pipeline until embed() is called", async () => {
    new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    expect(mockPipelineFactory).not.toHaveBeenCalled()
  })

  it("loads the pipeline exactly once across multiple embed() calls", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    await embedder.embed("hello", "query")
    await embedder.embed("world", "document")
    expect(mockPipelineFactory).toHaveBeenCalledTimes(1)
  })

  it("calls pipeline with the configured model and dtype", async () => {
    const embedder = new Embedder({ model: "BAAI/bge-small-en-v1.5", dtype: "q8" })
    await embedder.embed("test")
    expect(mockPipelineFactory).toHaveBeenCalledWith(
      "feature-extraction",
      "BAAI/bge-small-en-v1.5",
      { dtype: "q8" }
    )
  })
})

describe("Embedder – nomic task prefixes", () => {
  it("prepends 'search_query: ' for query role with nomic model", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    await embedder.embed("find me something", "query")
    expect(mockPipelineFn).toHaveBeenCalledWith(
      "search_query: find me something",
      expect.any(Object)
    )
  })

  it("prepends 'search_document: ' for document role with nomic model", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    await embedder.embed("read files from disk", "document")
    expect(mockPipelineFn).toHaveBeenCalledWith(
      "search_document: read files from disk",
      expect.any(Object)
    )
  })

  it("applies no prefix for non-nomic models", async () => {
    const embedder = new Embedder({ model: "BAAI/bge-small-en-v1.5", dtype: "fp32" })
    await embedder.embed("hello world", "query")
    expect(mockPipelineFn).toHaveBeenCalledWith("hello world", expect.any(Object))
  })
})

describe("Embedder – caching", () => {
  it("returns the same Float32Array instance on a repeated call", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    const a = await embedder.embed("cached text", "document")
    const b = await embedder.embed("cached text", "document")
    expect(a).toBe(b)
    // Pipeline invoked only once
    expect(mockPipelineFn).toHaveBeenCalledTimes(1)
  })

  it("treats the same text with different roles as distinct cache entries", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    const a = await embedder.embed("same text", "query")
    const b = await embedder.embed("same text", "document")
    expect(a).not.toBe(b)
    expect(mockPipelineFn).toHaveBeenCalledTimes(2)
  })

  it("clearCache() forces re-embedding on next call", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    await embedder.embed("text", "document")
    embedder.clearCache()
    await embedder.embed("text", "document")
    expect(mockPipelineFn).toHaveBeenCalledTimes(2)
  })
})

describe("Embedder – warmup", () => {
  it("warmup() triggers pipeline loading", async () => {
    const embedder = new Embedder({ model: "nomic-ai/nomic-embed-text-v1.5", dtype: "fp32" })
    await embedder.warmup()
    expect(mockPipelineFactory).toHaveBeenCalledTimes(1)
    // Subsequent embed() should not re-load
    await embedder.embed("hello", "query")
    expect(mockPipelineFactory).toHaveBeenCalledTimes(1)
  })
})
