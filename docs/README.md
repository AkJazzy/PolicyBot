
# PolicyBot - Internal Policy Document Assistant

## Overview
PolicyBot is an LLM-powered assistant that allows employees to query internal policy documents (e.g., HR policies, IT guidelines) using natural language. It uses LangChain for document loading, indexing, and retrieval.

## Features
- Upload and search internal PDFs
- Get cited answers with context
- Built using LangChain, OpenAI, and ChromaDB

## Requirements
- Python
- Streamlit
- LangChain
- OpenAI API Key

## How to Run
1. Place PDF files in the `data/` folder
2. Run the Streamlit app:
```bash
streamlit run app/main.py
```
