# GeneGPT: Tool-using LLMs for Bioinformatics

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **A modern implementation of GeneGPT concepts for answering complex bioinformatics questions using tool-enabled Large Language Models.**

Original paper: [GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information](https://arxiv.org/abs/2304.09667)

Original repository: [ncbi/GeneGPT](https://github.com/ncbi/GeneGPT/)

## 🧬 Overview

This project implements the concepts from the GeneGPT paper, leveraging advanced tool-use capabilities to answer complex bioinformatics questions. The system intelligently orchestrates calls to various NCBI APIs and bioinformatics tools, processes their outputs, and provides comprehensive, evidence-based answers.

## ✨ Key Features

### 🔧 **Comprehensive NCBI API Integration**
- **E-utilities Suite**: Search and retrieve data from PubMed, Gene, SNP, and OMIM databases
- **BLAST Integration**: Perform sequence similarity searches with support for multiple programs (blastn, blastp, blastx, tblastn, tblastx)
- **Intelligent Rate Limiting**: Built-in semaphore controls and request delays to respect NCBI guidelines

### 🛠️ **Advanced Tool Architecture**
- **Structured Output Parsing**: Robust parsing of JSON, XML, and text data from bioinformatics APIs
- **Tool Orchestration**: Smart selection and chaining of appropriate tools based on question context
- **Error Handling**: Comprehensive retry mechanisms and graceful degradation

### ⚡ **High-Performance Processing**
- **Concurrent Question Processing**: Multi-threaded execution with configurable worker pools
- **MLflow Integration**: Complete experiment tracking, model versioning, and performance metrics
- **Configurable Parameters**: Fine-tune LLM parameters, retry policies, and processing limits

### 🔄 **Multi-Provider LLM Support**
- **Azure OpenAI**: GPT-4.1, GPT-4.1-mini with tool calling capabilities
- **Local Models**: Ollama integration (e.g., Qwen3:4b) for privacy-sensitive workflows

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Azure OpenAI API key (for Azure provider) or Ollama installation (for local models)
- NCBI API key (recommended for higher rate limits)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd GeneGpt-Tool-Use
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or using uv
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   Required environment variables:
   ```env
   AZURE_OPENAI_API_KEY=your_azure_key_here
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   NCBI_API_KEY=your_ncbi_key_here  # Optional but recommended
   ```

4. **Start MLflow tracking server:**
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

### Basic Usage

**Run with tools enabled (recommended):**
```bash
python -m src.main \
  --dataset_path data/geneturing_small.json \
  --provider azure \
  --model gpt-4.1-mini \
  --output_path results/experiment_1.json \
  --tool_use true
```

**Run without tools (baseline comparison):**
```bash
python -m src.main \
  --dataset_path data/geneturing_small.json \
  --provider azure \
  --model gpt-4.1-mini \
  --output_path results/baseline_1.json \
  --tool_use false
```

## 📊 Available Datasets

The project includes several benchmark datasets for evaluation:

- **`geneturing_small.json`**: Subset of GeneTuring dataset for quick testing
- **`geneturing.json`**: Full GeneTuring benchmark dataset
- **`genehop.json`**: GeneHop dataset for multi-step reasoning evaluation

## 🛠️ Available Tools

### NCBI E-utilities
- **`esearch_ncbi`**: Search across Gene, SNP, and OMIM databases
- **`esummary_ncbi`**: Retrieve summary information for specific records
- **`efetch_ncbi`**: Fetch complete records in various formats

### BLAST Tools
- **`blast_put`**: Submit sequences for similarity searching
- **`blast_get`**: Retrieve BLAST results with customizable output formats

### Supported Databases
- **Gene**: Human gene information and annotations
- **SNP**: Single nucleotide polymorphism data
- **OMIM**: Online Mendelian Inheritance in Man
- **PubMed**: Biomedical literature database

## ⚙️ Configuration

Customize the behavior through `src/config.yaml`:

```yaml
llm_params:
  temperature: 0.7

mlflow_tracking_uri: "http://localhost:5000"

# Performance settings
MAX_WORKERS: 10        # Concurrent question processing
MAX_TURNS: 12          # Maximum tool-use iterations per question
MAX_RETRIES: 3         # API retry attempts
RETRY_DELAY: 5         # Seconds between retries
```

## 📈 Monitoring and Evaluation

### MLflow Integration

All experiments are automatically tracked with:
- **Model parameters** and configurations
- **Performance metrics** (accuracy, processing time)
- **Tool usage statistics**
- **Error rates** and failure modes

Access the MLflow UI at `http://localhost:5000` to:
- Compare model performance across different configurations
- Track tool usage patterns
- Monitor processing times and error rates
- Export results for further analysis

### Batch Evaluation Scripts

Use the provided scripts in `scripts/` for systematic evaluation:

```bash
# Run comprehensive evaluation across models and datasets
./scripts/geneturing_azure_gpt41_tools.sh
./scripts/genehop_azure_gpt41_mini_tools.sh
```