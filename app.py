import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()

if not google_api_key:
    st.error("GOOGLE_API_KEY is missing from your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key


# ---------------------------------------------------------
# Streamlit title
# ---------------------------------------------------------
st.title("Gemma Model Document Q&A")


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.

    Use the following pieces of context to answer the question.
    If you don't know the answer, just say you don't know.
    Don't try to make up an answer.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
)


# ---------------------------------------------------------
# Create vector database
# ---------------------------------------------------------
def vector_embedding():

    if "vectors" not in st.session_state:

        # -------------------------------------------------
        # Google Gemini Embeddings
        # FIXED: embedding-001 -> gemini-embedding-001
        # -------------------------------------------------
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        # -------------------------------------------------
        # Load PDF files from pdfs folder
        # -------------------------------------------------
        st.session_state.loader = PyPDFDirectoryLoader(
            "./pdfs"
        )

        st.session_state.docs = st.session_state.loader.load()

        # Check whether PDFs were found
        if not st.session_state.docs:
            st.error(
                "No PDF files found in the ./pdfs folder."
            )
            st.stop()

        # -------------------------------------------------
        # Split documents into chunks
        # -------------------------------------------------
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
        )

        # -------------------------------------------------
        # Create FAISS vector store
        # -------------------------------------------------
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


# ---------------------------------------------------------
# Button to create vector store
# ---------------------------------------------------------
if st.button("Documents Embedding"):

    with st.spinner("Creating vector store..."):

        try:
            vector_embedding()

            st.success(
                "Vector Store DB is Ready!"
            )

        except Exception as e:
            st.error(
                f"Error while creating vector store: {e}"
            )


# ---------------------------------------------------------
# User question
# ---------------------------------------------------------
prompt1 = st.text_input(
    "Enter Your Question From Documents"
)


# ---------------------------------------------------------
# Question answering
# ---------------------------------------------------------
if prompt1:

    if "vectors" not in st.session_state:

        st.warning(
            "Please create the Vector Store first."
        )

        st.stop()

    # -----------------------------------------------------
    # Create document chain
    # -----------------------------------------------------
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # -----------------------------------------------------
    # Get retriever
    # -----------------------------------------------------
    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 3}
    )

    # -----------------------------------------------------
    # Create retrieval chain
    # -----------------------------------------------------
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # -----------------------------------------------------
    # Start timer
    # -----------------------------------------------------
    start = time.process_time()

    try:

        # -------------------------------------------------
        # Get response
        # -------------------------------------------------
        response = retrieval_chain.invoke(
            {
                "input": prompt1
            }
        )

        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------
        st.subheader("Answer")

        st.write(
            response["answer"]
        )

        # -------------------------------------------------
        # Response time
        # -------------------------------------------------
        st.write(
            f"Response time: "
            f"{time.process_time() - start:.2f} seconds"
        )

        # -------------------------------------------------
        # Show relevant documents
        # -------------------------------------------------
        with st.expander(
            "Document Similarity Search"
        ):

            for i, doc in enumerate(
                response["context"]
            ):

                st.write(
                    f"Document {i + 1}:"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "--------------"
                )

    except Exception as e:

        st.error(
            f"Error while generating answer: {e}"
        )