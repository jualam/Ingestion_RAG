# Healthcare RAG LLM

A research project exploring retrieval-augmented generation (RAG) for healthcare transcription data, with emphasis on patient privacy and personally identifiable information (PII) handling.

The project uses LangChain, OpenAI models, and Pinecone vector search to ingest medical transcription text, retrieve relevant context, and evaluate model behavior across direct, masked, and hidden-query test cases.

## Project Contents

- `ingestion.py` - loads and embeds medical transcription data into a Pinecone index.
- `pii_ingestion.py` - ingests PII-focused transcription data for privacy experiments.
- `main.py`, `openAiChain.py` - run RAG-based question answering workflows.
- `performance.py` - evaluates generated answers using accuracy, factuality, and semantic similarity metrics.
- `fine_tuning_dataset.py`, `query_fine_tune.py` - utilities for fine-tuning dataset preparation and testing.
- `*_results*.txt` - experiment outputs and evaluation logs.

## Setup

This project uses Python 3.11 and Pipenv.

```bash
pipenv install
pipenv shell
```

Create a `.env` file with the required API keys and index names:

```env
OPENAI_API_KEY=
INDEX_NAME=
PII_INDEX_NAME=
```

## Basic Usage

Ingest transcription data:

```bash
python ingestion.py
```

Run a RAG query workflow:

```bash
python main.py
```

Run evaluation experiments:

```bash
python performance.py
```

## Research Scope

This repository is intended for experimentation with healthcare RAG pipelines, privacy-aware query handling, and evaluation of model responses over clinical transcription data. It is not intended for clinical deployment or medical decision-making.
