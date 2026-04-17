import type { PluginOptions } from "./config.js"

type FeatureExtractionPipeline = (
  text: string,
  options: { pooling: string; normalize: boolean }
) => Promise<{ data: Float32Array | number[] }>

/** Models that require a task-type prefix before input text. */
const PREFIX_MODELS: Record<string, { query: string; document: string }> = {
  "nomic-ai/nomic-embed-text-v1.5": {
    query: "search_query: ",
    document: "search_document: ",
  },
  "nomic-ai/nomic-embed-text-v1": {
    query: "search_query: ",
    document: "search_document: ",
  },
}

function getPrefix(
  modelId: string,
  role: "query" | "document"
): string {
  // Exact match
  if (PREFIX_MODELS[modelId]) return PREFIX_MODELS[modelId][role]
  // Partial match (e.g. custom nomic fine-tunes)
  for (const [key, prefixes] of Object.entries(PREFIX_MODELS)) {
    if (modelId.includes(key.split("/")[1])) return prefixes[role]
  }
  return ""
}

export class Embedder {
  private modelId: string
  private dtype: PluginOptions["dtype"]
  private pipePromise: Promise<FeatureExtractionPipeline> | null = null
  /** Cache: "<role>:<text>" → Float32Array */
  private cache = new Map<string, Float32Array>()

  constructor(opts: Pick<PluginOptions, "model" | "dtype">) {
    this.modelId = opts.model
    this.dtype = opts.dtype
  }

  private getOrLoadPipeline(): Promise<FeatureExtractionPipeline> {
    if (!this.pipePromise) {
      this.pipePromise = import("@huggingface/transformers").then(
        ({ pipeline }) =>
          pipeline("feature-extraction", this.modelId, {
            dtype: this.dtype,
          }) as Promise<FeatureExtractionPipeline>
      )
    }
    return this.pipePromise
  }

  /** Warm up the model (downloads weights on first call). */
  async warmup(): Promise<void> {
    await this.getOrLoadPipeline()
  }

  async embed(
    text: string,
    role: "query" | "document" = "document"
  ): Promise<Float32Array> {
    const cacheKey = `${role}:${text}`
    const cached = this.cache.get(cacheKey)
    if (cached) return cached

    const pipe = await this.getOrLoadPipeline()
    const prefix = getPrefix(this.modelId, role)
    const result = await pipe(prefix + text, { pooling: "mean", normalize: true })
    const embedding = Float32Array.from(result.data)

    this.cache.set(cacheKey, embedding)
    return embedding
  }

  /** Evict cached embeddings (e.g. when tools change). */
  clearCache(): void {
    this.cache.clear()
  }
}
