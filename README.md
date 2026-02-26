# Pittsburgh RAG Pipeline (ANLP Assignment 2)

This project builds a Retrieval-Augmented Generation (RAG) system for Pittsburgh-focused questions using four document collections:

- **A**: General and historical information
- **B**: Regulations and finance documents
- **C**: Dynamic events data
- **D**: Culture, food, and sports content

The pipeline combines:

- Query Router using `Llama-3.1-8B-Instruct` through Hugging Face Inference API
- Dense retrieval (FAISS + `BAAI/bge-m3` embeddings)
- Sparse retrieval (`bm25s` to implement BM25)
- Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`)
- Final answer generation through Hugging Face Inference API(`Qwen/Qwen2.5-14B-Instruct`)

## Repository Layout

- `data/raw/`: Raw source documents
- `data/processed/`: Processed documents, chunks, FAISS indexes, BM25 indexes
- `scripts/`: Core pipeline code (retrieval, reranking, routing, scraping, chunking)
- `*.ipynb`: Experiment and development notebooks
- `answers*.json`: Generated answer files

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirement.txt
```

`requirement.txt` includes retrieval, reranking, scraping, and notebook-related packages used in this repo.

3. Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token
```

## Quick Start

Use the main pipeline from Python:

```python
from scripts.rag_pipeline import RAGPipeline

pipe = RAGPipeline(verbose=True)
result = pipe.run("What are popular museums in Pittsburgh?")

print(result.answer)
print(result.collection)
```

To search all collections directly (without query routing):

```python
from scripts.rag_pipeline_all_collections import RAGPipelineAllCollections

pipe = RAGPipelineAllCollections(verbose=True)
result = pipe.run("What events are happening this weekend in Pittsburgh?")

print(result.answer)
```
## Notebooks Testing Order 

1. scraper.ipynb
2. chunker.ipynb
3. embedding.ipynb
4. sparse_search.ipynb
5. reranker.ipynb
6. query_router.ipynb
7. rag_pipeline.ipynb


## Notes

- Preprocessed data and indexes are already present in `data/processed/`.
- If you rebuild data, run the corresponding scripts under `scripts/chunkers/`, `scripts/build_index.py`, and `scripts/build_bm25.py`.
- Most development and experiments are documented in the notebooks.
