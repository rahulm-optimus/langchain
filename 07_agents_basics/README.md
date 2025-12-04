# Lesson 7: Agents Basics

## 🎯 Learning Objectives

- Understand what AI agents are
- Learn the agent execution loop
- Create your first LangChain agent
- Understand agent types
- Use agents with tools

## 🤖 What is an AI Agent?

An **agent** is an autonomous entity that uses an LLM to decide which actions to take and in what order. Unlike chains (which follow a predefined sequence), agents can make decisions dynamically based on user input.

### Key Characteristics:
1. **Autonomous** - Makes its own decisions
2. **Tool-using** - Can use external tools/APIs
3. **Iterative** - Can take multiple steps
4. **Goal-oriented** - Works toward solving a task

## 🔄 The Agent Loop

```
1. Receive user input
2. Think: What should I do?
3. Act: Use a tool or respond
4. Observe: See the result
5. Repeat until task is complete
```

### Example Flow:
```
User: "What's the weather in Paris and how do I say 'hello' in French?"

Agent thinks: I need to check weather and translate
→ Uses weather tool for Paris
→ Observes: "20°C, Sunny"
→ Uses translation tool
→ Observes: "Bonjour"
→ Combines information and responds
```

## 🏗️ Agent Components

### 1. **Agent (The Brain)**
- Decides what to do next
- Powered by an LLM
- Uses reasoning to solve problems

### 2. **Tools (The Hands)**
- External capabilities
- Examples: search, calculator, APIs
- Agent can use multiple tools

### 3. **Agent Executor**
- Runs the agent loop
- Handles tool execution
- Manages the conversation

## 📚 Types of Agents

### 1. **Zero-shot React Agent**
- Best for: General purpose tasks
- Features: Uses ReAct (Reasoning + Acting) framework
- When to use: Default choice for most cases

### 2. **Conversational React Agent**
- Best for: Multi-turn conversations
- Features: Has memory of conversation history
- When to use: Chatbots and interactive apps

### 3. **OpenAI Functions Agent**
- Best for: Structured tool usage
- Features: Uses OpenAI's function calling
- When to use: With GPT-3.5/4 for reliable tool use

### 4. **Structured Chat Agent**
- Best for: Multiple input tools
- Features: Can handle complex tool inputs
- When to use: Tools with multiple parameters

## 🛠️ ReAct Framework

ReAct = **Reasoning** + **Acting**

The agent alternates between:
- **Thought**: Reasoning about what to do
- **Action**: Taking an action with a tool
- **Observation**: Seeing the result

### Example:
```
Question: What is 25% of 200?

Thought: I need to calculate 25% of 200
Action: Calculator
Action Input: 200 * 0.25
Observation: 50
Thought: I now know the answer
Final Answer: 50
```

## 🎯 Building Your First Agent

### Basic Steps:
1. Import necessary modules
2. Create tools for the agent
3. Initialize the LLM
4. Create the agent
5. Create agent executor
6. Run the agent

### Simple Example:
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Define tools
tools = [...]

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run
response = agent_executor.invoke({"input": "Your question"})
```

## 💡 Best Practices

1. **Keep Tools Simple**: Each tool should do one thing well
2. **Clear Tool Descriptions**: Agent relies on descriptions
3. **Use Temperature 0**: For more reliable agent behavior
4. **Handle Errors**: Tools can fail, handle gracefully
5. **Limit Max Iterations**: Prevent infinite loops
6. **Verbose Mode**: Use for debugging

## 🎓 Code Examples

The `example.py` demonstrates:
- Creating a basic agent
- Using built-in tools
- Agent with custom tools
- Different agent types
- Handling agent errors

## 🐛 Common Issues

### Issue: Agent goes in loops
**Solution**: Use better tool descriptions, lower temperature

### Issue: Agent doesn't use tools
**Solution**: Improve tool descriptions, check tool registration

### Issue: Too many iterations
**Solution**: Set max_iterations parameter

### Issue: Agent makes up answers
**Solution**: Use temperature=0, better prompts

## 🔍 Agent vs Chain

| Feature | Chain | Agent |
|---------|-------|-------|
| Execution | Sequential | Dynamic |
| Decision Making | Predetermined | Autonomous |
| Tool Usage | Fixed | Adaptive |
| Complexity | Lower | Higher |
| Use Case | Known workflow | Unknown workflow |

## 🎯 When to Use Agents

✅ **Use Agents when:**
- You don't know the exact steps needed
- Tasks require tool usage
- Need dynamic problem solving
- Building autonomous systems

❌ **Don't use Agents when:**
- Workflow is fixed and known
- No tool usage needed
- Predictability is critical
- Cost/latency is a concern

## 🚀 Success Criteria

You're ready for the next lesson when you can:
- ✅ Explain what an agent is
- ✅ Understand the agent execution loop
- ✅ Create a basic agent with tools
- ✅ Choose the right agent type
- ✅ Debug agent behavior

## 🎓 Next Lesson

In Lesson 8, we'll explore Tools and Toolkits in depth, learning about built-in tools and how to use them effectively!
