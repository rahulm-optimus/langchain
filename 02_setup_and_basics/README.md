# Lesson 2: Setup and Basics

## 🎯 Learning Objectives

- Set up a Python environment for LangChain
- Install required dependencies
- Configure API keys
- Write your first LangChain code
- Understand basic LangChain workflow

## 🛠️ Setup Steps

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### 3. Set Up API Keys

Create a `.env` file in your project root:
```
OPENAI_API_KEY=sk-your-api-key-here
```

**How to get an OpenAI API key:**
1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to API keys section
4. Create a new secret key
5. Copy and save it securely

## 📚 Basic Concepts

### LangChain Components

1. **LLMs (Language Models)**
   - Text-in, text-out
   - Used for completion tasks

2. **Chat Models**
   - Messages-in, messages-out
   - Used for conversations

3. **Prompt Templates**
   - Reusable prompt structures
   - Variable substitution

4. **Output Parsers**
   - Structure LLM outputs
   - Convert text to Python objects

## 🔧 Basic Workflow

```
Input → Prompt Template → LLM → Output Parser → Result
```

## 💻 Your First LangChain Program

The `example.py` file demonstrates:
- Loading environment variables
- Creating an LLM instance
- Making a simple completion request
- Using chat models
- Working with prompt templates

## 🎓 Exercises

Complete the exercises in `exercises.py` to practice:
1. Setting up your environment
2. Making basic LLM calls
3. Using different models
4. Working with prompt templates

## 🔍 Key Functions to Know

- `ChatOpenAI()` - Create a chat model instance
- `invoke()` - Send a request to the model
- `PromptTemplate()` - Create reusable prompts
- `load_dotenv()` - Load environment variables

## 🐛 Common Issues

### Issue: Import errors
**Solution**: Ensure all packages are installed with correct versions

### Issue: API key errors
**Solution**: Check `.env` file exists and key is valid

### Issue: Rate limiting
**Solution**: Use a valid API key with credits

## 📝 Best Practices

1. **Always use environment variables** for API keys
2. **Never commit** `.env` files to version control
3. **Use virtual environments** to isolate dependencies
4. **Test with simple examples** before complex ones
5. **Monitor API usage** and costs

## 🎯 Success Criteria

You're ready for the next lesson when you can:
- ✅ Set up a Python environment
- ✅ Install LangChain packages
- ✅ Configure API keys securely
- ✅ Make a successful LLM call
- ✅ Use basic prompt templates

## 🚀 Next Lesson

In Lesson 3, we'll dive deeper into working with different LLM models and understanding their capabilities!
