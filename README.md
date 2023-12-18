# CDRC Semantic Search System

## Overview

The CDRC Semantic Search System is a project designed to enhance the search capabilities of the Centre for Consumer Data Research (CDRC) data catalogue. The goal is to implement a semantic search approach that goes beyond traditional keyword-based searches, providing users with more accurate and relevant results.

## Features

- **Semantic Search:** Utilizes advanced natural language processing techniques to understand the meaning behind user queries, enabling a more intuitive and precise search experience.

## System Architecture

The CDRC Semantic Search System follows a modular architecture. Below is a detailed diagram illustrating the stages and components required for the system:

```mermaid
graph TB

subgraph cluster_UI
  UI[User Interface]
  UIProc[User Input Processing]
end

subgraph cluster_Semantic
  SP[Semantic Processor]
  SA[Semantic Analysis]
end

subgraph cluster_Backend
  DB(Vector Database)
  LLM[Large Language Model]
  DCS[Data Catalog Service]
  RCD[Retrieve Catalog Data]
end

subgraph cluster_Results
  SR[Search Results]
end

UI --> UIProc
UIProc --> SP
SP --> SA
SA --> DB
SA --> LLM
DB --> DCS
LLM --> DCS
DCS --> RCD
RCD --> SR
```

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

   Edit the `config` json files to customize settings such as API keys, or model settings.

4. Run the system using a DVC pipeline.

   ```bash
   dvc repro
   ```

---

**Note:** The CDRC Semantic Search System is an ongoing project, and we appreciate your feedback and support in making it a valuable tool for researchers at the Centre for Consumer Data Research.
