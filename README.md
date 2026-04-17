# opencode-tool-filter

An [opencode](https://opencode.ai) plugin that **semantically filters MCP tools** to reduce context sent to the LLM, using local embedding inference via `@huggingface/transformers`.

## How it works

Two complementary strategies run on every request:

**1. Tool definition filtering** (primary — fires every request)

Before each LLM call, the plugin scores every tool definition against the current user query using cosine similarity. Tools that score below `minSimilarity` have their description and parameters blanked out — the LLM sees the tool name but can't use what it doesn't understand. Tools in `alwaysInclude` are never touched.

**2. Historical output pruning** (secondary — grows more effective over time)

Past tool call outputs are also scored against the current query. Irrelevant ones are replaced with `[pruned: not relevant to current query]`, keeping the conversation structure intact while reducing accumulated context.

```
User message
    ↓
chat.message hook  →  store query
    ↓
tool.definition hook (per tool)
    ├── embed query + tool description
    ├── cosine similarity < minSimilarity?
    └── yes → blank description + parameters
    ↓
experimental.chat.messages.transform hook
    ├── score past tool outputs against query
    └── replace irrelevant outputs with placeholder
    ↓
LLM receives filtered context
```

## Quick start

```bash
npm install @abelfodil/opencode-tool-filter
```

Add to `opencode.json`:

```json
{
  "plugin": ["@abelfodil/opencode-tool-filter"]
}
```

On first run the embedding model (~270 MB) is downloaded and cached by Hugging Face. Subsequent starts are instant.

## Configuration

All options are optional (defaults shown):

```json
["@abelfodil/opencode-tool-filter", {
  "enabled": true,

  // Embedding model (any HuggingFace feature-extraction model)
  "model": "nomic-ai/nomic-embed-text-v1.5",

  // Model weight dtype — "fp32" | "fp16" | "q8" | "q4" | "q4f16"
  // q8 / q4 are faster and smaller at a slight accuracy cost
  "dtype": "fp32",

  // Minimum cosine similarity to keep a tool definition or past output
  // Lower = keep more tools, higher = filter more aggressively
  "minSimilarity": 0.3,

  // For historical output pruning: keep at most N past outputs regardless of score
  "topK": 10,

  // Tool names / prefixes that are never filtered
  // Supports exact names ("bash") and MCP server prefixes ("mcp__myserver")
  "alwaysInclude": []
}]
```

### Tuning tips

- Start with `"minSimilarity": 0.3` and raise it if the LLM is missing tools it should use, or lower it if filtering is too aggressive.
- Add core tools to `alwaysInclude` to ensure they're always available regardless of the query: `["bash", "read", "edit"]`
- Use `"dtype": "q8"` for faster startup and lower memory usage with minimal accuracy loss.

### Choosing a different embedding model

Any HuggingFace `feature-extraction` model works:

```json
["@abelfodil/opencode-tool-filter", {
  "model": "BAAI/bge-small-en-v1.5",
  "dtype": "q8"
}]
```

Task-type prefixes (`search_query:` / `search_document:`) are applied automatically for the `nomic-ai/nomic-embed-*` family.

## Caveats

- **First request** after startup may be slightly slower while the embedding model loads.
- **Built-in opencode tools** (bash, read, edit, …) go through filtering too — add them to `alwaysInclude` if needed.
- Tool definition filtering uses only `minSimilarity` (not `topK`) since tools are scored one at a time. `topK` applies only to historical output pruning.

## License

MIT
