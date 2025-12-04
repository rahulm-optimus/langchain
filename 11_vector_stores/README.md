# Lesson 11: Vector Stores and Embeddings

## 🎯 Learning Objectives

- Understand embeddings and vector stores
- Implement semantic search
- Work with document loaders
- Build retrieval systems
- Optimize vector search

## 🎨 What are Embeddings?

**Embeddings** convert text into numerical vectors that capture semantic meaning. Similar concepts have similar vectors.

## 📊 Vector Stores

Vector stores efficiently store and search embeddings:
- **Chroma** - Easy to use, local
- **FAISS** - Fast, Facebook AI
- **Pinecone** - Cloud-based, scalable
- **Weaviate** - GraphQL API

## 🔍 Semantic Search

Unlike keyword search, semantic search finds **meaning**:
- "car" matches "automobile", "vehicle"
- Understands context and intent
- Better for Q&A and RAG

## 💡 Basic Workflow

1. **Load Documents** - Read files/data
2. **Split Text** - Break into chunks
3. **Create Embeddings** - Convert to vectors
4. **Store in Vector DB** - Index for search
5. **Query** - Find similar content

See `example.py` for implementation examples!
