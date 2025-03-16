# utils.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders  import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def load_qa_chain():
    # 1. Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. Load existing vector store
    vectorstore = FAISS.load_local(
        folder_path="vectorstore",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # Required for FAISS loading
    )
    
    # 3. Set up conversation memory
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # 4. Define prompt template
    template = """
    You are a helpful assistant that answers questions about Mir Ali. Use the following context and chat history to provide a gentle and informative response. 
    If the context doesn't provide the answer, politely say you don't have enough information.

    Context: {context}

    Chat History: {chat_history}

    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    
    # 5. Initialize Groq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    
    # 6. Create RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain