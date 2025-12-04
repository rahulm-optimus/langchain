# Lesson 12: Retrieval Agents (RAG)

## 🎯 Learning Objectives

- Build RAG (Retrieval-Augmented Generation) systems
- Combine vector search with agents
- Create Q&A systems over documents
- Implement document chat
- Optimize retrieval performance

## 🎨 What is RAG?

**RAG** = Retrieval-Augmented Generation

Instead of relying only on LLM's training data:
1. **Retrieve** relevant information from documents
2. **Augment** the prompt with retrieved context
3. **Generate** answers based on actual documents

## 🏗️ RAG Architecture

```
User Question
    ↓
Embed Question
    ↓
Search Vector Store
    ↓
Retrieve Relevant Docs
    ↓
Combine with Question
    ↓
LLM Generates Answer
    ↓
Response to User
```

## 🎯 Use Cases

- **Document Q&A** - Ask questions about PDFs
- **Knowledge Base** - Company documentation
- **Research Assistant** - Academic papers
- **Code Search** - Find code examples
- **Customer Support** - Policy documents

## 💡 Building RAG Systems

### Step 1: Prepare Documents
```python
from langchain.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PDFLoader("document.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

### Step 2: Create Vector Store
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

### Step 3: Create Retrieval Agent
```python
from langchain.agents import create_retrieval_agent

agent = create_retrieval_agent(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    tools=[...]
)
```

## 🎓 Advanced Techniques

### 1. **Hybrid Search**
Combine semantic + keyword search

### 2. **Re-ranking**
Improve retrieval quality

### 3. **Multi-Query**
Generate multiple search queries

### 4. **Metadata Filtering**
Filter by document properties

## 💡 Best Practices

1. **Chunk Size** - Balance context and precision
2. **Overlap** - Ensure continuity between chunks
3. **Top K** - Retrieve relevant number of docs
4. **Source Attribution** - Show where answers come from
5. **Handle No Results** - Graceful fallback

## 🚀 Production Tips

- Cache embeddings
- Index optimization
- Monitor retrieval quality
- Update documents regularly
- Handle large document sets

See `example.py` for complete RAG implementation!

---

## 🎉 Congratulations!

You've completed the LangChain learning path! You now know how to:
- Set up and use LangChain
- Work with LLMs and prompts
- Build chains and workflows
- Create AI agents
- Use tools effectively
- Build custom tools
- Implement RAG systems
- Deploy production applications

### Next Steps:
1. Build your own projects
2. Explore LangGraph for complex workflows
3. Learn LangServe for deployment
4. Join the community
5. Keep learning and experimenting!

**Happy building! 🚀**
