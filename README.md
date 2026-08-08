# ContextIQ — Agentic RAG Knowledge Assistant

An **Agentic Retrieval-Augmented Generation (RAG)** application built with **LangChain, LangGraph, FAISS, OpenAI, and Streamlit**.

ContextIQ allows users to ask questions against an indexed document corpus and uses an agentic workflow to retrieve relevant information and generate grounded answers. When the indexed documents do not contain sufficient information, the agent can use **Wikipedia** as an external knowledge source.

---

## 🚀 Key Features

- **Document ingestion**
  - Load content from web URLs
  - Load PDF documents
  - Load text files
- **Document chunking**
  - Recursive character-based text splitting
  - Configurable chunk size and overlap
- **Semantic retrieval**
  - OpenAI embeddings
  - FAISS vector store
  - Similarity-based document retrieval
- **Agentic RAG**
  - LangChain agent
  - Retriever tool for the indexed corpus
  - Wikipedia search tool for general knowledge
  - Tool selection based on the question
- **LangGraph orchestration**
  - Retriever node
  - Agentic answer-generation node
  - Explicit workflow/state management
- **Streamlit UI**
  - Interactive question-answer interface
  - Retrieved source-document inspection
  - Response-time measurement
  - Recent-search history

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   User Question     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LangGraph         │
                    │   Workflow          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Retriever Node   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FAISS Vector     │
                    │       Store         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Agentic RAG Agent  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Retriever Tool  │        │ Wikipedia Tool  │
        │ Indexed Corpus  │        │ General         │
        │                 │        │ Knowledge       │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    OpenAI LLM       │
                    │   Answer Generation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Grounded Answer    │
                    └─────────────────────┘
```

---

## 🔄 RAG Workflow

```text
Question
   │
   ▼
Retriever
   │
   ▼
Agentic Answer Generation
   │
   ├── Indexed documents
   │
   └── Wikipedia when required
   │
   ▼
Final Answer
```

---

## 📚 Document Processing

Documents are loaded using LangChain document loaders and processed with `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk Size    : 500
Chunk Overlap : 50
```

Supported ingestion paths include:

- Web URLs
- PDF directories
- TXT files

---

## 🧠 Vector Search

ContextIQ uses:

- **OpenAI Embeddings**
- **FAISS**
- LangChain retriever

```text
Documents
    ↓
OpenAI Embeddings
    ↓
FAISS
    ↓
Retriever
```

---

## 🤖 Agentic RAG

The agent has two tools.

### 1. Retriever Tool

Searches the indexed document corpus and returns relevant passages.

The agent is designed to use the indexed document corpus first for questions related to the user's documents.

### 2. Wikipedia Tool

Provides general knowledge when the indexed corpus does not contain sufficient information.

The agent is instructed not to repeatedly call an unavailable tool and not to invent facts.

### Agent Behavior

The agent is designed to:

- Use the indexed corpus first
- Use Wikipedia when necessary
- Avoid repeatedly calling unavailable tools
- Avoid inventing facts
- Clearly state when sufficient information is unavailable

---

## 🔗 LangGraph

The application uses `RAGState` to carry information through the workflow.

Current state:

```python
class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
```

Current graph:

```text
START
  │
  ▼
retriever
  │
  ▼
responder
  │
  ▼
END
```

---

## 🖥️ Streamlit Application

The Streamlit application initializes:

1. OpenAI LLM
2. Document processor
3. Vector store
4. LangGraph workflow

The RAG system is cached using Streamlit's `cache_resource`.

Users can:

- Enter a question
- Submit the query
- View the generated answer
- Inspect retrieved source documents
- See response time
- Review the last three searches

---

## ⚙️ Technology Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | OpenAI |
| Agent Framework | LangChain |
| Workflow Orchestration | LangGraph |
| Embeddings | OpenAI Embeddings |
| Vector Database | FAISS |
| Document Processing | LangChain Document Loaders |
| External Knowledge | Wikipedia |
| State Management | Pydantic |
| Language | Python |

---

## 📁 Project Structure

```text
rag-demo/
│
├── src/
│   ├── config/
│   │   └── config.py
│   │
│   ├── document_ingestion/
│   │   └── document_processor.py
│   │
│   ├── vectorstore/
│   │   └── vectorstore.py
│   │
│   ├── graph_builder/
│   │   └── graph_builder.py
│   │
│   ├── node/
│   │   ├── nodes.py
│   │   └── reactnode.py
│   │
│   └── state/
│       └── rag_state.py
│
├── streamlit_app.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── .python-version
├── .env
└── README.md
```

---

## 🔧 Configuration

The application currently uses:

```text
LLM Model       : openai:gpt-4o
Chunk Size      : 500
Chunk Overlap   : 50
```

The default document sources are configured in `config.py`.

---

## ▶️ Running the Application

The project uses **uv** for Python environment and dependency management.

### 1. Clone the repository

```cmd
git clone <repository-url>
cd rag-demo
```

### 2. Initialize the uv project

This repository already contains a `pyproject.toml`. If you are setting up
the project from scratch and no `pyproject.toml` exists, initialize it with:

```cmd
uv init
```

> **Note:** For the current repository, `pyproject.toml` already exists, so
> `uv init` is normally **not required**.

### 3. Create the virtual environment

```cmd
uv venv
```

This creates the local `.venv` environment.

### 4. Activate the virtual environment

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install project dependencies

Install the dependencies listed in `requirements.txt`:

```cmd
uv add -r requirements.txt
```

Install the Jupyter/IPython kernel support used during development:

```cmd
uv add ipykernel
```

Then synchronize the environment:

```cmd
uv sync
```

### 6. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

**Never commit `.env` or API keys to source control.**

### 7. Run the Streamlit application

```cmd
streamlit run streamlit_app.py
```

The application will start and Streamlit will display the local URL,
typically:

```text
http://localhost:8501
```

### 8. Optional: run Streamlit through uv

You can also run the application without manually activating `.venv`:

```cmd
uv run streamlit run streamlit_app.py
```

### Complete Windows setup sequence

For a fresh checkout of this repository:

```cmd
git clone <repository-url>
cd rag-demo
uv venv
.venv\Scripts\activate
uv add -r requirements.txt
uv add ipykernel
uv sync
streamlit run streamlit_app.py
```

### Development reset

If you need to recreate the environment:

```cmd
deactivate
rmdir /s /q .venv
uv venv
.venv\Scripts\activate
uv sync
streamlit run streamlit_app.py
```

> **Important:** Do not use `uv init` as a normal reset command. The project
> already has a `pyproject.toml`; `uv venv` and `uv sync` are the appropriate
> commands for recreating the environment.


---

## 💡 Example Questions

```text
What are LLM-powered autonomous agents?
```

```text
What are the key components of an AI agent?
```

```text
How does retrieval help an LLM-based application?
```

You can also ask general-knowledge questions where the agent may use Wikipedia when the indexed corpus does not provide sufficient information.

---

## 🔐 Security

- Keep API keys outside source control.
- Use environment variables for secrets.
- Never commit `.env` files containing credentials.
- Rotate any API key that has been accidentally exposed.

---

## 🚧 Current Scope

This is a **demo implementation of Agentic RAG** intended to demonstrate:

- RAG fundamentals
- Semantic retrieval
- Vector search
- Agent tool usage
- LangGraph orchestration
- External knowledge fallback
- LLM-based answer generation
- Streamlit-based AI application development

---

## 🔮 Potential Future Enhancements

- Conversational memory
- Multi-turn RAG
- Metadata filtering
- Hybrid search
- Reranking
- Citation generation
- Document upload from the UI
- Persistent vector databases
- Evaluation pipelines
- Retrieval quality metrics
- LLM evaluation
- Guardrails
- Observability and tracing
- Multi-agent workflows
- Enterprise document connectors
- Authentication and authorization
- Production-grade API layer

---
