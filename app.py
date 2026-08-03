
import os
import warnings
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_classic.chains.retrieval import (
    create_retrieval_chain,
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key is None:
    groq_api_key = st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------
# Initialize LLM
# ---------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.title("Conversational RAG Chatbot")

session_id = st.text_input(
    "Session ID",
    value="default_session"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# ---------------------------------------------------
# Chat History Store
# ---------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(
    session_id: str,
) -> BaseChatMessageHistory:

    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    return st.session_state.store[session_id]


# ---------------------------------------------------
# Process PDF
# ---------------------------------------------------

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save uploaded pdf temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_pdf_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split Documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        split_docs = splitter.split_documents(documents)

        # Create Vector Store
        vectorstore = FAISS.from_documents(
            split_docs,
            embeddings,
        )

        retriever = vectorstore.as_retriever()

    st.success("PDF Processed Successfully!")

    # ---------------------------------------------------
    # Contextualize Prompt
    # ---------------------------------------------------

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Given the chat history and the latest user question,
rewrite the latest question into a standalone question.

Do NOT answer it.

Only rewrite it.
""",
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            ("human", "{input}"),
        ]
    )

    # ---------------------------------------------------
    # QA Prompt
    # ---------------------------------------------------

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context,
say:

"I don't know based on the provided documents."

Context:
{context}
""",
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            ("human", "{input}"),
        ]
    )

    # ---------------------------------------------------
    # Chains
    # ---------------------------------------------------

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt,
    )

    document_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    conversational_rag = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # ---------------------------------------------------
    # User Question
    # ---------------------------------------------------

    user_question = st.text_input(
        "Ask a question"
    )

    if user_question:

        response = conversational_rag.invoke(
            {
                "input": user_question
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
        )

        st.subheader("Answer")

        st.write(response["answer"])

        # ---------------------------------------------

        with st.expander("Retrieved Documents"):

            for i, doc in enumerate(response["context"], start=1):

                st.markdown(f"### Document {i}")

                st.write(doc.page_content)

                st.divider()

        # ---------------------------------------------

        with st.expander("Chat History"):

            history = get_session_history(session_id)

            for message in history.messages:

                st.write(message)

else:

    st.info("Please upload a PDF.")