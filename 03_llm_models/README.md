# Lesson 3: LLM Models

## 🎯 Learning Objectives

- Understand different types of language models
- Work with various LLM providers
- Configure model parameters effectively
- Understand token limits and pricing
- Choose the right model for your use case

## 📚 Types of Language Models

### 1. **Completion LLMs (Legacy)**
- Text-in, text-out
- Good for: completion tasks, text generation
- Examples: GPT-3 base models

### 2. **Chat Models (Recommended)**
- Messages-in, message-out
- Good for: conversations, structured interactions
- Examples: GPT-3.5-turbo, GPT-4, Claude

### 3. **Embedding Models**
- Text-in, vector-out
- Good for: semantic search, similarity
- Examples: text-embedding-ada-002

## 🔧 Popular LLM Providers

### OpenAI
- **GPT-3.5-turbo**: Fast, affordable, general purpose
- **GPT-4**: More capable, slower, expensive
- **GPT-4-turbo**: Faster GPT-4, larger context

### Anthropic
- **Claude 3 Haiku**: Fast and cost-effective
- **Claude 3 Sonnet**: Balanced performance
- **Claude 3 Opus**: Most capable

### Open Source
- **Llama 2**: Meta's open-source model
- **Mistral**: High-performance open model

## ⚙️ Key Model Parameters

### Temperature (0.0 - 2.0)
- **0.0**: Deterministic, consistent outputs
- **0.7**: Balanced (default for most tasks)
- **1.0+**: Creative, varied outputs

### Max Tokens
- Maximum length of the response
- Includes both prompt and completion
- Different for each model

### Top P (0.0 - 1.0)
- Alternative to temperature
- Controls diversity of output

### Frequency Penalty (-2.0 - 2.0)
- Reduces repetition
- Positive values discourage repeated tokens

### Presence Penalty (-2.0 - 2.0)
- Encourages topic diversity
- Positive values encourage new topics

## 💰 Understanding Tokens and Cost

### What is a Token?
- ~4 characters in English
- ~3/4 of a word
- 1 token ≈ 0.75 words

### Example:
```
"Hello, how are you?" = ~5 tokens
"The quick brown fox jumps" = ~5 tokens
```

### Cost Optimization Tips:
1. Use GPT-3.5-turbo for simple tasks
2. Limit max_tokens appropriately
3. Use streaming for long responses
4. Cache frequently used prompts

## 🎯 Choosing the Right Model

### Use GPT-3.5-turbo when:
- You need fast responses
- The task is straightforward
- Cost is a concern
- Real-time interactions required

### Use GPT-4 when:
- Complex reasoning required
- Accuracy is critical
- Longer context needed
- Handling nuanced tasks

### Use Claude when:
- You need longer context windows
- Safety and alignment are priorities
- Detailed analysis required

## 🔍 Model Comparison

| Model | Speed | Cost | Context | Best For |
|-------|-------|------|---------|----------|
| GPT-3.5-turbo | Fast | Low | 16K | General tasks |
| GPT-4 | Slow | High | 8K | Complex reasoning |
| GPT-4-turbo | Medium | Medium | 128K | Long documents |
| Claude 3 Haiku | Fast | Low | 200K | Fast processing |
| Claude 3 Opus | Slow | High | 200K | Complex analysis |

## 💻 Code Examples

The `example.py` file demonstrates:
- Using different models
- Configuring parameters
- Streaming responses
- Token counting
- Model comparison

## 🔥 Best Practices

1. **Start Simple**: Begin with GPT-3.5-turbo
2. **Test Parameters**: Experiment with temperature
3. **Monitor Costs**: Track token usage
4. **Use Streaming**: For better UX with long responses
5. **Set Max Tokens**: Prevent runaway costs
6. **Handle Errors**: Implement retry logic

## 🐛 Common Issues

### Issue: Token limit exceeded
**Solution**: Reduce input size or use models with larger context

### Issue: Slow responses
**Solution**: Use faster models or implement streaming

### Issue: High costs
**Solution**: Optimize prompts, use cheaper models, cache results

### Issue: Inconsistent outputs
**Solution**: Lower temperature for more deterministic results

## 🎓 Practice Exercises

In `exercises.py`, you'll practice:
1. Using different models
2. Adjusting parameters
3. Token counting
4. Model selection
5. Cost optimization

## 🎯 Success Criteria

You're ready for the next lesson when you can:
- ✅ Choose appropriate models for different tasks
- ✅ Configure model parameters effectively
- ✅ Understand token usage and costs
- ✅ Implement streaming responses
- ✅ Handle model errors gracefully

## 🚀 Next Lesson

In Lesson 4, we'll learn about Prompt Templates and how to create dynamic, reusable prompts!
