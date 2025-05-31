# GeneGPT: Tool-using LLMs to answer bioinformatics questions

Original paper: https://arxiv.org/abs/2304.09667

Original code: https://github.com/ncbi/GeneGPT/

## Overview

This project is a implementation of the concepts presented in the GeneGPT paper, focusing on leveraging advanced tool-use capabilities to answer complex bioinformatics questions. It intelligently interacts with various APIs and processes their outputs to provide comprehensive answers.

## Features

*   **Structured Output Parsing**: Reliably parses and utilizes structured data (e.g., JSON, XML) returned by LLM API's and various bioinformatics 
*   **Multithreading for Concurrency**: Efficiently handles multiple API calls and data processing tasks simultaneously. 
*   **NCBI API Integration**: Seamlessly interacts with a suite of National Center for Biotechnology Information (NCBI) APIs, including E-utilities (for literature searches via PubMed, gene information via Entrez Gene, etc.), BLAST (for sequence similarity searches)

*   **MLflow for Tracking, Tracing, and Metrics**: Integrates with MLflow for robust tracking of experiments, model parameters, code versions, results, and performance metrics.
