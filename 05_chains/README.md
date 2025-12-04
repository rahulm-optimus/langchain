# Lesson 5: Chains - Complete Guide

## 🎯 Learning Objectives

- Understand all LangChain chain types
- Master LCEL (modern pipe syntax)
- Build sequential and conditional workflows
- Implement chains with memory
- Use specialized chains for specific tasks
- Create custom chains

## 🔗 What are Chains?

**Chains** combine multiple components (LLMs, prompts, parsers, tools) into a single workflow. They allow you to:
- Execute steps in sequence or parallel
- Transform data between steps
- Reuse common patterns
- Build complex AI applications
- Add memory and state management

---

## 📊 Chain Types Overview

| Chain Type | Usage | Status | Complexity |
|------------|-------|--------|------------|
| **LCEL (Pipe `\|`)** | 🟢 **85%** | ✅ Recommended | ⭐⭐ |
| **LLMChain** | 🟡 10% | ⚠️ Legacy | ⭐ |
| **Sequential Chains** | 🟡 3% | ⚠️ Legacy | ⭐⭐ |
| **Router Chains** | 🔴 1% | ⚠️ Rarely used | ⭐⭐⭐⭐ |
| **Specialized Chains** | 🟡 1% | ✅ For specific tasks | ⭐⭐⭐ |

---

## 🚀 **1. LCEL - LangChain Expression Language** ⭐ **RECOMMENDED**

Modern, pipe-based syntax for building chains.

### **Basic LCEL Chain**
```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="Tell me about {topic}",
    input_variables=["topic"]
)

llm = OllamaLLM(model="llama3.2")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "Python"})
```

### **Multi-Step LCEL Chain**
```python
chain = (
    prompt1 
    | llm 
    | StrOutputParser() 
    | prompt2 
    | llm 
    | StrOutputParser()
)

result = chain.invoke({"text": "AI is transforming the world"})
```

### **LCEL with Streaming**
```python
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## 📦 **2. LLMChain** (Legacy - Still Works)

Basic chain combining prompt + LLM.

```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

prompt = PromptTemplate(
    template="Translate {text} to {language}",
    input_variables=["text", "language"]
)

llm = OllamaLLM(model="llama3.2")
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.invoke({
    "text": "Hello",
    "language": "Spanish"
})
```

---

## 🔄 **3. Sequential Chains**

Run multiple chains in order, passing outputs between them.

### **SimpleSequentialChain**
Output of one chain → Input of next (single variable)

```python
from langchain.chains import SimpleSequentialChain, LLMChain

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = overall_chain.invoke("Write a poem")
```

### **SequentialChain**
Multiple inputs/outputs between chains

```python
from langchain.chains import SequentialChain

chain1 = LLMChain(
    llm=llm, 
    prompt=prompt1, 
    output_key="story"
)

chain2 = LLMChain(
    llm=llm, 
    prompt=prompt2, 
    output_key="summary"
)

overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["story", "summary"]
)

result = overall_chain.invoke({"topic": "space exploration"})
```

---

## 🧭 **4. Router Chains**

Route inputs to different chains based on conditions.

### **MultiPromptChain**
```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains import ConversationChain

# Define destination chains
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for physics questions",
        "prompt_template": "Answer this physics question: {input}"
    },
    {
        "name": "math",
        "description": "Good for math questions",
        "prompt_template": "Solve this math problem: {input}"
    },
    {
        "name": "history",
        "description": "Good for history questions",
        "prompt_template": "Answer this history question: {input}"
    }
]

destination_chains = {}
for p_info in prompt_infos:
    prompt = PromptTemplate(
        template=p_info["prompt_template"],
        input_variables=["input"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[p_info["name"]] = chain

# Default chain
default_chain = ConversationChain(llm=llm, output_key="text")

# Create router
router_chain = MultiPromptChain(
    router_chain=LLMRouterChain.from_prompts(llm, prompt_infos),
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

result = router_chain.invoke("What is gravity?")
```

---

## 🔧 **5. TransformChain**

Apply custom Python functions between LLM calls.

```python
from langchain.chains import TransformChain, LLMChain

def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    return {"output_text": text.upper()}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["output_text"],
    transform=transform_func
)

# Combine with LLM chain
prompt = PromptTemplate(
    template="Analyze this text: {output_text}",
    input_variables=["output_text"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Chain them together
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[transform_chain, llm_chain],
    input_variables=["text"],
    output_variables=["text"]
)
```

---

## 📝 **6. Specialized Chains**

Pre-built chains for specific use cases.

### **ConversationChain** (with Memory)
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.invoke("Hi, my name is Alice")
conversation.invoke("What's my name?")  # Remembers context
```

### **LLMRequestsChain** (Fetch URL data)
```python
from langchain.chains import LLMRequestsChain

chain = LLMRequestsChain(llm=llm)
result = chain.invoke({
    "url": "https://api.example.com/data",
    "question": "What is the main topic?"
})
```

### **LLMMathChain** (Solve math problems)
```python
from langchain.chains import LLMMathChain

math_chain = LLMMathChain.from_llm(llm, verbose=True)
result = math_chain.invoke("What is 25 * 17 + 340?")
```

### **SQLDatabaseChain** (Query databases)
```python
from langchain.chains import SQLDatabaseChain
from langchain.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///mydb.db")

chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True
)

result = chain.invoke("How many users are there?")
```

### **RetrievalQA Chain** (RAG - Document Q&A)
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_chain.invoke("What does the document say about AI?")
```

---

## 🎯 **Common Use Cases**

### **1. Translation Pipeline**
```
English Text → Detect Language → Translate to Spanish → Verify Grammar → Format
```

### **2. Content Creation**
```
Topic → Research → Generate Outline → Write Draft → Edit → Format → Publish
```

### **3. Data Analysis**
```
Raw Data → Clean → Transform → Analyze → Summarize → Visualize
```

### **4. Customer Support Bot**
```
User Query → Classify Intent → Route to Expert Chain → Generate Response → Add Context
```

---

## 💡 **Best Practices**

1. ✅ **Use LCEL (`|`)** - Modern, recommended approach
2. ✅ **Keep chains simple** - One responsibility per chain
3. ✅ **Add error handling** - Validate between steps
4. ✅ **Test individually** - Verify each component
5. ✅ **Use streaming** - Better UX for long responses
6. ✅ **Add logging** - Debug complex workflows
7. ✅ **Monitor performance** - Track execution time
8. ✅ **Document flow** - Explain the workflow clearly

---

## 🚨 **Common Pitfalls**

❌ **Too many steps** - Break into smaller chains  
❌ **No error handling** - Chains can fail at any step  
❌ **Hardcoded values** - Use variables and configs  
❌ **Missing validation** - Check outputs between steps  
❌ **Poor memory management** - Token limits can break chains  

---

## 📚 **Quick Reference**

### **When to Use What?**

| Use Case | Chain Type |
|----------|------------|
| **Any new project** | LCEL (`\|`) |
| **Simple Q&A** | LCEL |
| **Multi-step processing** | LCEL or SequentialChain |
| **Conditional routing** | RouterChain or custom logic |
| **Chat with memory** | ConversationChain + Memory |
| **Math calculations** | LLMMathChain |
| **Database queries** | SQLDatabaseChain |
| **Document Q&A** | RetrievalQA |
| **Custom transformations** | TransformChain |

---

## 🎓 **Summary**

**Modern Approach (2024+):**
- Use **LCEL** (pipe syntax) for 90%+ of use cases
- Clean, flexible, supports streaming
- Easy to debug and maintain

**Legacy Chains:**
- Still work but less recommended
- Use if maintaining existing codebase
- Consider migrating to LCEL

**Specialized Chains:**
- Use for specific tasks (math, SQL, RAG)
- Save development time
- Well-tested implementations

See `example.py` for practical implementations of all chain types!
