# Quick Migration Guide

## From Hardcoded to Dynamic Model Configuration

### 🚨 **IMMEDIATE ACTION REQUIRED**

Your codebase has been updated to remove hardcoded models. **All existing code that uses `get_config()` without parameters will now fail** because no default model is set.

### ✅ **How to Fix Your Code**

#### **Before (OLD - Will now fail):**

```python
from src.utils.config import get_config
from src.llm_clients import OpenAIClient

# This will now raise an error because no model is specified
config = get_config()  # ❌ WILL FAIL
client = OpenAIClient(config)
```

#### **After (NEW - Working):**

```python
from src.utils.config import get_config
from src.llm_clients import OpenAIClient

# Specify model and tier explicitly
config = get_config(model="gpt-4.1-mini", tier="tier2")  # ✅ WORKS
client = OpenAIClient(config)
```

---

### 🎯 **Recommended Models for Your Tier 2 Setup**

#### **Option 1: Cost-Effective (Recommended)**

```python
# Use gpt-4o-mini - cheapest option ($0.15/$0.60 per 1M tokens)
config = get_config(model="gpt-4o-mini", tier="tier2")
```

#### **Option 2: Balanced Performance**  

```python
# Use gpt-4.1-mini - good balance of cost and capability
config = get_config(model="gpt-4.1-mini", tier="tier2") 
```

#### **Option 3: Maximum Capability**

```python
# Use gpt-4o - most capable but expensive
config = get_config(model="gpt-4o", tier="tier2")
```

#### **Option 4: Reasoning Tasks**

```python
# Use o1-mini - best for reasoning but most expensive
config = get_config(model="o1-mini", tier="tier2")
```

---

### 🔧 **Your Tier 2 Rate Limits Applied Automatically**

When you use the new system, these limits are automatically applied:

| Model | RPM | TPM | Daily Requests | Cost/1M tokens |
|-------|-----|-----|----------------|----------------|
| **gpt-4o-mini** | 1,800 | 1,800,000 | 288,000 | $0.15/$0.60 |
| **gpt-4.1-mini** | 4,500 | 1,800,000 | 648,000 | $0.40/$1.60 |
| **gpt-4o** | 4,500 | 405,000 | 648,000 | $2.50/$10.00 |
| **o1-mini** | 1,800 | 1,800,000 | 288,000 | $3.00/$12.00 |

*All values include 90% safety margins*

---

### 📋 **Update Checklist**

Find and update all occurrences of:

1. **✅ Config Loading:**

   ```python
   # OLD
   config = get_config()
   
   # NEW  
   config = get_config(model="gpt-4o-mini", tier="tier2")  # Choose your model
   ```

2. **✅ Experiment Scripts:**

   ```python
   # OLD
   from src.utils.config import get_config
   config = get_config()
   
   # NEW
   from src.utils.config import get_config
   config = get_config(model="gpt-4o-mini", tier="tier2")  # Cheapest option
   ```

3. **✅ Test Files:**

   ```python
   # OLD
   client = OpenAIClient(get_config())
   
   # NEW
   client = OpenAIClient(get_config(model="gpt-4o-mini", tier="tier2"))
   ```

---

### 🎯 **My Recommendation for You**

Based on your Tier 2 status, I recommend starting with **gpt-4o-mini**:

```python
# Use this in all your code for maximum cost savings
config = get_config(model="gpt-4o-mini", tier="tier2")
```

**Why gpt-4o-mini?**

- ✅ **Cheapest**: $0.15/$0.60 per 1M tokens (75% cheaper than gpt-4.1-mini)
- ✅ **Fast**: 1,800 RPM, 1.8M TPM rate limits  
- ✅ **Capable**: Latest OpenAI mini model with good performance
- ✅ **Batch Processing**: 1.8M token batch queue limit
- ✅ **Still high limits**: 288,000 requests per day

You can always switch models by changing just the model parameter:

```python
# Switch to gpt-4.1-mini for better performance when needed
config = get_config(model="gpt-4.1-mini", tier="tier2")

# Switch to gpt-4o for maximum capability on important tasks  
config = get_config(model="gpt-4o", tier="tier2")
```

---

### 🧪 **Test Your Migration**

Run this test to verify everything works:

```python
# Test script - save as test_migration.py
from src.utils.config import get_config
from src.llm_clients import OpenAIClient

def test_models():
    models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "o1-mini"]
    
    for model in models:
        try:
            print(f"Testing {model}...")
            config = get_config(model=model, tier="tier2")
            client = OpenAIClient(config)
            
            rate_limits = config.get("openai.rate_limits")
            print(f"  ✅ {model}: {rate_limits['requests_per_minute']:,} RPM")
            
        except Exception as e:
            print(f"  ❌ {model}: Error - {e}")
    
    print("✅ Migration test complete!")

if __name__ == "__main__":
    test_models()
```

The system is now properly configured for your Tier 2 setup with dynamic model switching!
