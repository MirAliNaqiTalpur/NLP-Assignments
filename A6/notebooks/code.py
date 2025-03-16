# Start with necessary imports
import os
from dotenv import load_dotenv
import torch
import json 
from langchain.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings 
from langchain.vectorstores import FAISS, Chroma
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BitsAndBytesConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load environment variables from .env file
load_dotenv()

# 1) Find all relevant sources about yourself

# List your personal documents
personal_docs = [
    "../data/linkedin_profile.pdf",
    "../data/personal_bio.txt",
    "../data/AIT_SIS_personal_info.txt"
]

# Load documents
documents = []
for doc_path in personal_docs:
    try:
        loader = None
        
        if doc_path.endswith('.pdf'):
            loader = PyMuPDFLoader(doc_path)
        elif doc_path.endswith('.txt'):
            loader = TextLoader(doc_path)
        
        if loader:
            documents.extend(loader.load())
            print(f"Successfully loaded: {doc_path}")
        else:
            print(f"No suitable loader found for: {doc_path}")
            
    except Exception as e:
        print(f"Error loading {doc_path}: {e}")


# Print document count
print(f"Loaded {len(documents)} documents")



# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, #1000
    chunk_overlap=200   #200
)
doc_chunks = text_splitter.split_documents(documents)
print(f"Created {len(doc_chunks)} document chunks")

# Set up embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": str(device)}
)

# Create vector store
vectorstore = FAISS.from_documents(
    documents=doc_chunks,
    embedding=embedding_model
)

# Save vectorstore locally
vectorstore.save_local("vectorstore")
print("Vector store saved successfully")

# 2) Design prompt template
template = """
I'm your friendly personal assistant, here to answer questions about myself(Mir Ali).
I'll provide detailed and accurate information based on the available documents about my background,
education, work experience, skills, and  interests..

Context information from my personal records:
{context}

Current conversation:
{chat_history}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

# 3) Explore different text generation models
# Option 1: Use HuggingFace model

# For local model
model_path = "../models/fastchat-t5-3b-v1.0"   
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    legacy=False,
    padding_side="left",
    truncation_side="right",
    model_max_length=512
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    device_map='auto'  
)

# Create pipeline
pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    padding=True,
    truncation=True,
    max_length=512,
    model_kwargs={
        "temperature": 0,
        "repetition_penalty": 1.5,
        "max_length": 512
    }
)

# Wrap in LangChain
local_llm = HuggingFacePipeline(pipeline = pipe)

# Option 2: Use Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
    )
    print("Groq LLM configured successfully")
else:
    groq_llm = None
    print("Warning: GROQ_API_KEY not found in environment variables")

# Define all models
models = {"FastChat-T5": local_llm}
if groq_llm:
    models["Groq-llama3-70b"] = groq_llm
    
    # Initialize results dictionary
results = {model_name: [] for model_name in models.keys()}

# Create the chain
def create_qa_chain(llm):
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    # Test with required questions
questions = [
    "How old are you?",
    "What is your highest level of education?",
    "What major or field of study did you pursue during your education?",
    "How many years of work experience do you have?",
    "What type of work or industry have you been involved in?",
    "Can you describe your current role or job responsibilities?",
    "What are your core beliefs regarding the role of technology in shaping society?",
    "How do you think cultural values should influence technological advancements?",
    "As a master's student, what is the most challenging aspect of your studies so far?",
    "What specific research interests or academic goals do you hope to achieve during your time as a master's student?"
]

# Test with all models
for model_name, llm in models.items():
    try:
        print(f"\n{'='*40}")
        print(f"Testing model: {model_name}")
        print(f"{'='*40}\n")
        
        qa_chain = create_qa_chain(llm)
        
        for i, question in enumerate(questions, start=1):
            try:
                response = qa_chain({"question": question})
                results[model_name].append({
                    "question_number": i,
                    "question": question,
                    "answer": response["answer"]
                })
                print(f"Q{i}: {question}")
                print(f"A{i}: {response['answer']}")
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                error_msg = f"Error processing question {i}: {e}"
                print(error_msg)
                results[model_name].append({
                    "question_number": i,
                    "question": question,
                    "answer": f"ERROR: {error_msg}"
                })
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")

# Save all results to JSON
with open("all_model_answers.json", "w") as f:
    json.dump(results, f, indent=2)

print("Testing complete. Results saved to all_model_answers.json")