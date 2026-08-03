# 📄 Conversational PDF Query Chatbot with Stored Chat History

A **Conversational Retrieval-Augmented Generation (RAG)** chatbot built using **Streamlit**, **LangChain**, **Groq Llama 3.3**, **Hugging Face Embeddings**, and **FAISS**. The application allows users to upload a PDF, ask questions about its content, and continue the conversation using persistent chat history.

---

## 🚀 Features

- 📄 Upload and query PDF documents.
- 💬 Conversational question answering with stored chat history.
- 🧠 History-aware retriever that understands follow-up questions.
- 🔍 Semantic search using Hugging Face embeddings and FAISS.
- ⚡ Fast inference using Groq's Llama 3.3 model.
- 🆔 Session-based conversation history.
- 📚 View retrieved document chunks used to generate answers.
- 🎨 Simple and interactive Streamlit interface.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (Llama 3.3 70B Versatile)
- **Framework:** LangChain
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Database:** FAISS
- **Document Loader:** PyPDFLoader
- **Memory:** RunnableWithMessageHistory & ChatMessageHistory

---

## 📂 Project Structure

```text
Conversational_Pdf_query_chatbot_with_storedhistory/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/Conversational_Pdf_query_chatbot_with_storedhistory.git

cd Conversational_Pdf_query_chatbot_with_storedhistory
```

### Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` File

```env
GROQ_API_KEY=your_groq_api_key
```

### Run the Application

```bash
streamlit run app.py
```

---

## 🔄 Workflow

```text
                 Upload PDF
                      │
                      ▼
                Load Document
                      │
                      ▼
             Split into Chunks
                      │
                      ▼
        Generate Embeddings (MiniLM)
                      │
                      ▼
             Store in FAISS Index
                      │
                      ▼
               User asks Question
                      │
                      ▼
         History Aware Retriever
        (Rewrites Follow-up Query)
                      │
                      ▼
         Retrieve Relevant Chunks
                      │
                      ▼
         Llama 3.3 Generates Answer
                      │
                      ▼
          Store Conversation History
```

---

## 🧠 LangChain Components Used

- PyPDFLoader
- RecursiveCharacterTextSplitter
- HuggingFaceEmbeddings
- FAISS
- create_history_aware_retriever
- create_stuff_documents_chain
- create_retrieval_chain
- RunnableWithMessageHistory
- ChatMessageHistory

---

## 💬 Example

**User**

> What is LangChain?

**Assistant**

> LangChain is an open-source framework for developing applications powered by large language models.

**User**

> Who created it?

**Assistant**

> LangChain was created by Harrison Chase.

The chatbot understands that **"it"** refers to **LangChain** because it uses conversation history.

---

## 🌟 Future Enhancements

- Multiple PDF upload support
- Persistent vector database (ChromaDB/Pinecone)
- Source page citations
- Streaming responses
- Authentication
- Chat export
- Cloud deployment

---

## 👨‍💻 Author

**Tanish Batra**

GitHub: https://github.com/Tanish326

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub!