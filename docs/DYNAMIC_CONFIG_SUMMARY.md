# Dynamic Model Configuration System

## Implementation Summary

### 🎯 **Problem Solved**

- **Removed hardcoded models** throughout the codebase
- **Implemented tier-based rate limits** for all OpenAI models
- **Dynamic configuration loading** based on model and tier
- **Support for Tier 2 rate limits** (your current tier)

### 📊 **Tier 2 Rate Limits by Model**

| Model | RPM | TPM | Batch Queue Limit | Cost (Input/Output per 1M tokens) |
|-------|-----|-----|-------------------|-----------------------------------|
| **gpt-4.1-mini** | 5,000 | 2,000,000 | 20,000,000 | $0.40 / $1.60 |
| **gpt-4o-mini** | 2,000 | 2,000,000 | 2,000,000 | $0.15 / $0.60 |
| **o1-mini** | 2,000 | 2,000,000 | 2,000,000 | $3.00 / $12.00 |
| **gpt-4o** | 5,000 | 450,000 | 1,350,000 | $2.50 / $10.00 |

*All values include 90% safety margins applied automatically*

---

### 🚀 **Usage Examples**

#### **Method 1: Dynamic Config Loading (Recommended)**

```python
from src.utils.config import get_config
from src.llm_clients import OpenAIClient

# Load gpt-4.1-mini with Tier 2 limits (recommended for cost-effectiveness)
config = get_config(model="gpt-4.1-mini", tier="tier2")
client = OpenAIClient(config)

# Switch to gpt-4o-mini for maximum cost savings
config = get_config(model="gpt-4o-mini", tier="tier2") 
client = OpenAIClient(config)

# Use o1-mini for reasoning tasks
config = get_config(model="o1-mini", tier="tier2")
client = OpenAIClient(config)

# Use gpt-4o for maximum capability
config = get_config(model="gpt-4o", tier="tier2")
client = OpenAIClient(config)
```

#### **Method 2: Direct Rate Limit Access**

```python
from src.utils.config import get_model_rate_limits

# Get rate limits for any model/tier combination
limits = get_model_rate_limits("gpt-4.1-mini", "tier2")
print(limits)
# Output: {'requests_per_minute': 4500, 'tokens_per_minute': 1800000, ...}
```

#### **Method 3: Configuration File Loading**

The system automatically loads from `config/models/{model}_{tier}.yaml` files:

- `config/models/gpt-4.1-mini_tier2.yaml`
- `config/models/gpt-4o-mini_tier2.yaml`
- `config/models/o1-mini_tier2.yaml`
- `config/models/gpt-4o_tier2.yaml`

---

### 📂 **Files Created/Modified**

#### **New Configuration Files:**

1. **`config/models/gpt-4.1-mini_tier2.yaml`** - Tier 2 config for gpt-4.1-mini
2. **`config/models/gpt-4o-mini_tier2.yaml`** - Tier 2 config for gpt-4o-mini  
3. **`config/models/o1-mini_tier2.yaml`** - Tier 2 config for o1-mini
4. **`config/models/gpt-4o_tier2.yaml`** - Tier 2 config for gpt-4o
5. **`config/USAGE_EXAMPLES.yaml`** - Comprehensive usage examples

#### **New Python Modules:**

1. **`src/utils/model_config.py`** - Model configuration manager
2. **`scripts/demo_model_config.py`** - Demo script for testing

#### **Modified Files:**

1. **`config/default.yaml`** - Removed hardcoded models, added placeholders
2. **`src/utils/config.py`** - Added model-specific config loading
3. **`src/llm_clients/openai_client.py`** - Removed hardcoded defaults
4. **`src/llm_clients/openai_batch_client.py`** - Removed hardcoded defaults
5. **`src/llm_clients/__init__.py`** - Added gpt-4o-mini, removed hardcoded defaults

---

### 🔧 **Key Features**

#### **1. Automatic Safety Margins**

- All rate limits automatically include 90% safety margins
- Prevents hitting actual API limits

#### **2. Dynamic Fallback Models**

- Each model has appropriate fallback models configured
- Automatic fallback pricing for unknown models

#### **3. Tier Support**

- Supports all OpenAI tiers (1-5) where applicable
- Defaults to Tier 2 (your current tier)

#### **4. Backwards Compatibility**

- Old `get_config()` calls still work (loads default.yaml)
- New `get_config(model="...", tier="...")` for dynamic loading

#### **5. Cost Optimization**

- Model-specific pricing information
- Batch processing limits included
- Cost estimation functions updated

---

### 💡 **Migration Guide**

#### **Old Hardcoded Approach:**

```python
# OLD - Always used gpt-4.1-mini with Tier 1 limits
config = get_config()  
client = OpenAIClient(config)
```

#### **New Dynamic Approach:**

```python
# NEW - Explicit model and tier selection
config = get_config(model="gpt-4.1-mini", tier="tier2")  # Your current setup
config = get_config(model="gpt-4o-mini", tier="tier2")   # Cheapest option
config = get_config(model="o1-mini", tier="tier2")       # Reasoning model
config = get_config(model="gpt-4o", tier="tier2")        # Most capable
```

---

### 🎯 **Recommendations for Your Tier 2 Setup**

#### **For Cost-Effectiveness:**

- **Primary**: `gpt-4o-mini` (cheapest: $0.15/$0.60 per 1M tokens)
- **Secondary**: `gpt-4.1-mini` (good balance: $0.40/$1.60 per 1M tokens)

#### **For Maximum Throughput:**

- **Primary**: `gpt-4.1-mini` (5,000 RPM, 2M TPM)
- **Secondary**: `gpt-4o` (5,000 RPM, 450K TPM)

#### **For Reasoning Tasks:**

- **Primary**: `o1-mini` (2,000 RPM, 2M TPM)
- **Backup**: `gpt-4.1-mini`

#### **For Maximum Capability:**

- **Primary**: `gpt-4o` (5,000 RPM, 450K TPM)
- **Backup**: `gpt-4.1-mini`

---

### 🧪 **Testing the New System**

#### **1. Run Demo Script:**

```bash
# Test specific model
python scripts/demo_model_config.py --model gpt-4o-mini

# Test all models
python scripts/demo_model_config.py --all

# Test different tier
python scripts/demo_model_config.py --model gpt-4.1-mini --tier tier2
```

#### **2. Quick Configuration Test:**

```python
from src.utils.config import get_config

# This should work without errors and show your Tier 2 limits
config = get_config(model="gpt-4.1-mini", tier="tier2")
print(config.get("openai.rate_limits"))
```

#### **3. Client Initialization Test:**

```python
from src.utils.config import get_config
from src.llm_clients import OpenAIClient

config = get_config(model="gpt-4o-mini", tier="tier2")  # Cheapest option
client = OpenAIClient(config)
print(f"Using model: {client.primary_model}")
print(f"Rate limits: {client.rate_limiter.get_current_usage()}")
```

---

### ✅ **Benefits Achieved**

1. **✅ No more hardcoded models** - All models are configurable
2. **✅ Tier 2 rate limits** - Proper limits for your current tier
3. **✅ Model flexibility** - Easy switching between models
4. **✅ Cost optimization** - Model-specific pricing and limits
5. **✅ Safety margins** - Automatic 90% rate limit safety
6. **✅ Backwards compatibility** - Existing code still works
7. **✅ Future-proof** - Easy to add new models and tiers

The system is now fully dynamic and will automatically apply the correct rate limits and configuration based on your model choice and tier level!
