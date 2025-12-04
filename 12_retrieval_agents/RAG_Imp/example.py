"""
RAG Implementation - Simple Learning Example
Clean version with efficient logging
"""
import os
import sys
import time
import signal
import atexit
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm

load_dotenv()

# Configuration
DOCUMENTS_PATH = "./documents"
CHROMA_PATH = "./chroma_db_new"
MODEL_NAME = os.getenv("model_2")
LOG_FILE = "rag_system.log"

# Setup Logging (100MB max, 3 backups)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=100*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(file_handler)

# Track start time
start_time = datetime.now()

def log_shutdown():
    """Log when app stops"""
    duration = datetime.now() - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"App stopped | Runtime: {hours}h {minutes}m {seconds}s")
    logger.info("="*50)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.warning("Interrupted by user (Ctrl+C)")
    log_shutdown()
    sys.exit(0)

# Register shutdown handlers
atexit.register(log_shutdown)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Log startup
logger.info("="*50)
logger.info(f"App started | Model: {MODEL_NAME}")


def load_documents():
    """Load PDF, TXT, and MD files"""
    print("📂 Loading documents...")
    
    pdf_loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    md_loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    
    docs = []
    try:
        docs.extend(pdf_loader.load())
        docs.extend(txt_loader.load())
        docs.extend(md_loader.load())
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        print(f"❌ {e}")
        return []
    
    logger.info(f"Loaded {len(docs)} documents")
    print(f"✅ {len(docs)} documents loaded\n")
    return docs


def split_documents(documents):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    print(f"✅ {len(chunks)} chunks created\n")
    return chunks


def create_vectorstore(chunks):
    """Create vector store with embeddings"""
    print(f"🔍 Creating vector store ({len(chunks)} chunks)...")
    logger.info(f"Vector store creation started ({len(chunks)} chunks)")
    
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma(
        collection_name="document_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    start = time.time()
    with tqdm(total=len(chunks), desc="Embedding", unit="chunk") as pbar:
        for i, chunk in enumerate(chunks):
            try:
                vectorstore.add_documents(documents=[chunk])
                pbar.update(1)
                
                # Log every 50 chunks
                if (i + 1) % 50 == 0:
                    elapsed = (time.time() - start) / 60
                    logger.info(f"Progress: {i+1}/{len(chunks)} | {elapsed:.1f}min")
                
            except Exception as e:
                logger.error(f"Failed at chunk {i+1}: {e}")
                raise
    
    duration = (time.time() - start) / 60
    logger.info(f"Vector store created in {duration:.1f}min")
    print(f"\n✅ Done in {duration:.1f} minutes\n")
    
    return vectorstore


def load_existing_vectorstore():
    """Load existing vector store"""
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma(
        collection_name="document_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    logger.info("Loaded existing vector store")
    print("✅ Vector store loaded\n")
    return vectorstore


def setup_rag_chain(vectorstore):
    """Setup RAG chain"""
    llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=4096)
    
    template = """Answer based ONLY on the context below. If you don't know, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content.strip()}"
            for doc in docs
        ])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain initialized")
    print("✅ RAG ready\n")
    return rag_chain, retriever


def query_documents(rag_chain, retriever, question):
    """Query and get answer"""
    print(f"\n❓ {question}")
    logger.info(f"Query: {question}")
    
    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    
    print(f"\n💡 {answer}\n")
       
    logger.info(f"Retrieved {len(source_docs)} docs | Answer: {len(answer)} chars")
    
    return answer, source_docs


def main():
    """Main RAG workflow"""
    print("🚀 RAG System\n")
    
    # Check if rebuild needed
    rebuild = False
    if os.path.exists(CHROMA_PATH):
        choice = input("📊 Vector store exists. Rebuild? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            rebuild = True
            logger.info("Rebuilding vector store")
            import shutil
            try:
                shutil.rmtree(CHROMA_PATH)
                print("✅ Old vector store deleted\n")
            except Exception as e:
                logger.error(f"Delete failed: {e}")
                print(f"❌ {e}")
                return
    
    # Build or load vector store
    if rebuild or not os.path.exists(CHROMA_PATH):
        documents = load_documents()
        if not documents:
            logger.error("No documents found")
            print("❌ No documents in ./documents folder")
            return
        
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks)
    else:
        vectorstore = load_existing_vectorstore()
    
    # Setup RAG
    rag_chain, retriever = setup_rag_chain(vectorstore)
    
    # Interactive loop
    print("="*60)
    print("💬 Ask questions about your documents (type 'exit' to quit)")
    print("="*60 + "\n")
    
    while True:
        question = input("🤔 Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            logger.info("User exited")
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            query_documents(rag_chain, retriever, question)
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            logger.error(f"Query error: {e}")
            print(f"❌ {e}")


if __name__ == "__main__":
    main()
