# CDRC Semantic Search System

## Overview

The CDRC Semantic Search System is a project designed to enhance the search capabilities of the Centre for Consumer Data Research (CDRC) data catalogue. The goal is to implement a semantic search approach that goes beyond traditional keyword-based searches, providing users with more accurate and relevant results.

## Features

- **Semantic Search:** Embeds documents using OpenAI which are stored on Pinecone, allowing for semantic querying using cosine similarity.

- **Retrieval Augmented Generation:** Generates responses using GPT 3.5 turbo to explain the relevance of retrieved datasets.

## System Architecture

The CDRC Semantic Search System follows a standard Retrieval Augmented Generation (RAG) architecture:

![Credit to Heiko Hotz (https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7)](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Jq9bEbitg1Pv4oASwEQwJg.png)

_Credit to Heiko Hotz (https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7)_

## Installation

To get started with the CDRC Semantic Search System, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/cjber/cdrc-semantic-search.git
   ```

2. Install dependencies:

_With pip:_

```bash
cd cdrc-semantic-search
pip install -r requirements.txt
```

_With pdm:_

```bash
cd cdrc-semantic-search
pdm install
```

3. Configure the system:

   Edit the `config/config.toml` file to customize settings such as API keys, or model settings.

4. Run the system using a DVC pipeline.

   ```bash
   dvc repro
   ```

> NOTE: This requires a Pinecone database and access to the CDRC catalogue.
