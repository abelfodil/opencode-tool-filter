import { describe, it, expect } from "vitest"
import { PluginOptionsSchema } from "../src/config.js"

describe("PluginOptionsSchema", () => {
  it("applies all defaults on empty input", () => {
    const opts = PluginOptionsSchema.parse({})
    expect(opts.enabled).toBe(true)
    expect(opts.model).toBe("nomic-ai/nomic-embed-text-v1.5")
    expect(opts.dtype).toBe("fp32")
    expect(opts.topK).toBe(10)
    expect(opts.minSimilarity).toBe(0.3)
    expect(opts.alwaysInclude).toEqual([])
  })

  it("accepts all valid dtype values", () => {
    for (const dtype of ["fp32", "fp16", "q8", "q4", "q4f16"] as const) {
      expect(PluginOptionsSchema.parse({ dtype }).dtype).toBe(dtype)
    }
  })

  it("rejects an unknown dtype", () => {
    expect(() => PluginOptionsSchema.parse({ dtype: "int8" })).toThrow()
  })

  it("rejects topK < 1", () => {
    expect(() => PluginOptionsSchema.parse({ topK: 0 })).toThrow()
  })

  it("rejects minSimilarity > 1", () => {
    expect(() => PluginOptionsSchema.parse({ minSimilarity: 1.1 })).toThrow()
  })

  it("accepts partial overrides and preserves other defaults", () => {
    const opts = PluginOptionsSchema.parse({ topK: 5, alwaysInclude: ["bash"] })
    expect(opts.topK).toBe(5)
    expect(opts.alwaysInclude).toEqual(["bash"])
    expect(opts.model).toBe("nomic-ai/nomic-embed-text-v1.5")
  })

  it("accepts enabled: false", () => {
    expect(PluginOptionsSchema.parse({ enabled: false }).enabled).toBe(false)
  })
})
