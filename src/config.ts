import { z } from "zod"

export const PluginOptionsSchema = z.object({
  enabled: z.boolean().default(true),

  /** HuggingFace model ID for computing embeddings. */
  model: z.string().default("nomic-ai/nomic-embed-text-v1.5"),

  /** Model weight dtype — "fp32" | "fp16" | "q8" | "q4" | "q4f16" */
  dtype: z.enum(["fp32", "fp16", "q8", "q4", "q4f16"]).default("fp32"),

  /** Keep only the N most-similar tool results per request. */
  topK: z.number().int().positive().default(10),

  /** Minimum cosine similarity [0–1] to keep a tool result. 0 = keep topK regardless. */
  minSimilarity: z.number().min(0).max(1).default(0.3),

  /** Tool names / prefixes whose outputs are never pruned. */
  alwaysInclude: z.array(z.string()).default([]),
})

export type PluginOptions = z.infer<typeof PluginOptionsSchema>
