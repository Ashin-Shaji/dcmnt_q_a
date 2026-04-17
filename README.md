# 📄 Intelligent Document Q&A System

An advanced **Retrieval-Augmented Generation (RAG)** application that enables interactive querying of local documents. Powered by **Gemini 1.5 Pro** and **Google Text-Embeddings**, this tool allows users to upload PDF, DOCX, or TXT files and receive context-aware answers grounded strictly in the document's content.

---

## 🏗️ System Architecture

The application implements a classic **RAG (Retrieval-Augmented Generation)** pipeline, ensuring the AI only answers based on the provided context to prevent hallucinations.



### 1. Ingestion & Extraction Layer
* **Multi-Format Parsing:** Utilizes `PyMuPDF (fitz)` for PDF extraction, `docx2txt` for Word documents, and standard I/O for text files.
* **Storage Management:** Automatically manages a local `pdfs/` directory for persistent file selection and provides tools for batch-clearing session data.

### 2. Semantic Embedding Layer
* **Text Chunking:** Implements a **Fixed-Size Chunking** strategy (default 1000 characters) to ensure that text fits within the model's context window while maintaining semantic meaning.
* **Vectorization:** Converts text chunks into high-dimensional vectors using `models/text-embedding-004`. These embeddings represent the mathematical "meaning" of the text.

### 3. Retrieval & Reasoning Layer
* **Cosine Similarity:** When a user asks a question, the system vectorizes the query and performs a **Dot Product (Cosine Similarity)** calculation against the document embeddings to find the most relevant paragraph.
* **Prompt Engineering:** Injects the retrieved context into a specialized system prompt that instructs the LLM to remain strictly objective and within the provided document scope.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM Core** | Google Gemini 1.5 Pro |
| **Embeddings** | Google Generative AI Embeddings (`text-embedding-004`) |
| **Framework** | Streamlit |
| **Vector Math** | NumPy |
| **Document Processing** | PyMuPDF (fitz) & docx2txt |

---

## 🧠 Logic & Workflow

1.  **Upload & Select:** The user uploads a file to the sidebar or selects an existing one from the "Vault."
2.  **Proceed (Embedding):** Upon clicking "Proceed," the system triggers the extraction and embedding pipeline.
    
3.  **Querying:** The user inputs a question. The system finds the specific chunk with the highest similarity score.
4.  **Generation:** The relevant chunk and the question are sent to Gemini 1.5 Pro to generate a formatted, readable response.

---

## 📊 Technical Specifications

### Semantic Search Logic
The retrieval engine uses **Numpy-accelerated similarity scoring** to identify the context:
```python
similarities = [np.dot(question_embedding, text_embedding) for text_embedding in text_embeddings]
most_relevant_index = np.argmax(similarities)
```

### Prompt Guardrails
To ensure accuracy, the system uses a **Restricted Context Prompt**:
> *"You are a document analyzer and you will not talk irrelevant to the given context, only based on this Context: {context} answer this Question: {question}"*

### Persistence & Caching
The system uses Streamlit's `session_state` to store `text_embeddings` and `text_chunks`. This allows users to ask multiple questions about the same document without re-running the expensive embedding process (Token Efficiency).
