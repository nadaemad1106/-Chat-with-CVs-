# 📄 AI HR Matcher (RAG System)

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** system tailored for HR departments to chat with and analyze CVs efficiently.

## Core Technologies
* **ChromaDB**: For vector storage and similarity search.
* **Sentence Transformers**: Using `multilingual-e5-small` for high-quality text embeddings.
* **Google Gemini 2.5 Flash**: For intelligent response generation and job title validation.
* **LangChain**: To orchestrate the document loading, splitting, and retrieval chain.
* **Streamlit**: For the interactive web interface.

## Key Features
* [cite_start]**Automatic Name Extraction**: Identifies candidate names directly from PDF content[cite: 1, 42, 87].
* **Job Position Validation**: A built-in HR Auditor that detects and rejects fake or non-standard job titles.
* **Anti-Injection Security**: Guardrails to prevent users from bypassing HR rules (e.g., asking for jokes or 'PASSED' status).
* **Top-K Adaptive Retrieval**: Dynamically filters chunks based on similarity thresholds and specific candidate detection.

## Setup

1. Create virtual environment:
   python -m venv venv

2. Activate:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create .env file:
   GOOGLE_API_KEY=your_api_key_here

5. Run:
   Streamlit run app.py

