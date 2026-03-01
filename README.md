# soul.py v2.0 Demo — RAG + RLM Hybrid

Live demo for [soul.py v2.0-rag-rlm branch](https://github.com/menonpg/soul.py/tree/v2.0-rag-rlm).

Three retrieval modes:
- **Auto** — router classifies each query, dispatches to RAG or RLM
- **RAG only** — semantic vector search (focused queries)
- **RLM only** — recursive synthesis (exhaustive queries)

**v0.1 demo:** https://soul.themenonlab.com
**v1.0 demo:** https://soul-v1.themenonlab.com
**v2.0 demo:** this app

## Railway env vars required

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | your Anthropic key |
| `QDRANT_URL` | Qdrant Cloud URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `AZURE_EMBEDDING_ENDPOINT` | Azure OpenAI endpoint |
| `AZURE_EMBEDDING_KEY` | Azure OpenAI key |
