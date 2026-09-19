# Gemma Document Q&A

A Retrieval-Augmented Generation (RAG) app that lets you ask questions about your own PDF documents. Built with Streamlit, LangChain, Groq, and Google Gemini embeddings.

## Features

- Upload PDFs to a local `pdfs/` folder and index them into a vector store
- Fast LLM responses via Groq (`openai/gpt-oss-20b`)
- Semantic search powered by Google Gemini embeddings (`gemini-embedding-001`)
- FAISS vector database for similarity search
- Simple Streamlit UI with response time and source-document view

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://www.langchain.com/) — RAG orchestration
- [Groq](https://groq.com/) — LLM inference
- [Google Generative AI](https://ai.google.dev/) — embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — vector store
- [PyPDF](https://pypi.org/project/pypdf/) — PDF loading

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/gemma-doc-qa.git
cd gemma-doc-qa
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (see `.env.example`):

```
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
```

- Get a Groq API key: https://console.groq.com/keys
- Get a Google API key: https://aistudio.google.com/app/apikey

### 5. Add your PDFs

Place the PDF files you want to query inside the `pdfs/` folder.

### 6. Run the app

```bash
streamlit run app.py
```

## Usage

1. Click **"Documents Embedding"** to build the vector store from the PDFs in `pdfs/`.
2. Once it's ready, type your question in the text box.
3. View the answer, response time, and the source document chunks used to generate it.

## Project Structure

```
Gemma/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── pdfs/               # PDF files to be indexed
├── .env.example        # Sample environment variables
└── .gitignore
```

## Notes

- The `.env` file and `nenv/` (or `venv/`) virtual environment folder are excluded from version control via `.gitignore` — never commit real API keys.
- The vector store is rebuilt each session; consider persisting it to disk if you need it to survive restarts.

## License

MIT
