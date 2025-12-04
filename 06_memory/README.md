# Lesson 6: Memory (Modern Approach)

## 🎯 Learning Objectives

- Add memory to conversations using modern patterns
- Implement multi-session chat history
- Use `RunnableWithMessageHistory` (LCEL)
- Manage conversation context
- Handle session isolation

## 🧠 What is Memory?

**Memory** allows LLMs to remember previous interactions and maintain context across conversations.

## 🆕 Modern Approach (2024+)

LangChain now uses **`RunnableWithMessageHistory`** instead of legacy memory classes.

### ✅ Modern Pattern (Recommended)

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store conversations
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create prompt with history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Chain with memory
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Use with session ID
response = chain_with_history.invoke(
    {"input": "Hi, I'm Alice"},
    config={"configurable": {"session_id": "user_123"}}
)
```

### ❌ Legacy Pattern (Deprecated)

```python
# Don't use these anymore!
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
```

## 📚 Key Concepts

### 1. **Session-Based History**
Each user/conversation has unique session ID
```python
# User 1
chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "user_1"}}
)

# User 2 (separate memory)
chain_with_history.invoke(
    {"input": "My name is Bob"},
    config={"configurable": {"session_id": "user_2"}}
)
```

### 2. **Message Placeholders**
Use `MessagesPlaceholder` in prompts
```python
MessagesPlaceholder(variable_name="history")
```

### 3. **In-Memory Storage**
`InMemoryChatMessageHistory` stores messages in RAM (lost on restart)

### 4. **Clearing History**
```python
history = get_session_history("user_123")
history.clear()
```

## 🎯 Use Cases

- **Chatbots** - Remember user context
- **Customer Support** - Track conversation flow
- **Tutoring Systems** - Remember what was taught
- **Multi-User Apps** - Isolated sessions per user

## 💡 Best Practices

1. **Use Session IDs** - Unique per user/conversation
2. **Clear Old Data** - Reset when conversation ends
3. **Limit History** - Keep last N messages to save tokens
4. **Persist Storage** - Use database for production (not in-memory)
5. **Handle Errors** - Gracefully handle memory failures

## 🔄 Migration Guide

**Old (Legacy):**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

**New (Modern):**
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

See `example.py` for complete implementations!
