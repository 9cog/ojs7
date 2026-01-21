# 🤖 Copilot Prompt Seed — AI Inference Enforcement

## System Goals

- Build a fully working AI inference engine using llama.cpp, node-llama-cpp, or production BERT models
- Avoid any mock, stub, or placeholder logic in AI agent code
- Ensure all fallback paths still use real inference engines (e.g., quantized models)
- Prioritize functionality over performance during initial implementation
- Enforce production-grade AI standards throughout the codebase

## Copilot Behavior Directives

### Core AI Implementation Rules
- Always suggest production-grade code for AI inference modules
- Do not use `return "TODO"`, `pass`, or mock classes for AI agents
- Use llama.cpp, node-llama-cpp, or BERT models for all completions and inference tasks
- If a model is unavailable, fallback to a quantized variant (e.g., ggml-model-q4_0.bin)
- Prefer native integration (C++, WASM, or Node bindings) over placeholder logic
- In PRs or features related to inference, assume llama.cpp is the required backend

### Production Quality Standards
- All AI agent modules must invoke a real inference engine
- Any fallback must still produce real model output
- Linting/CI will block PRs with mock or incomplete inference logic
- Environment variables must be configured for model paths and settings
- Error handling must degrade gracefully to simpler but real AI models

### AI Architecture Patterns
- Initialize models once and cache for performance
- Use async/await patterns for non-blocking inference
- Implement proper resource management for GPU/CPU usage
- Add configuration for model paths, context lengths, and quantization
- Support both local and remote inference endpoints

## Implementation Examples

### llama.cpp Integration (Node.js)
```js
const { LlamaModel, LlamaContext, LlamaChatSession } = require('node-llama-cpp');

class ProductionAIEngine {
    constructor(config) {
        this.modelPath = config.modelPath || './models/llama-2-7b-chat.q4_0.gguf';
        this.contextSize = config.contextSize || 2048;
        this.model = null;
        this.context = null;
    }

    async initialize() {
        this.model = new LlamaModel({
            modelPath: this.modelPath
        });
        this.context = new LlamaContext({
            model: this.model,
            contextSize: this.contextSize
        });
    }

    async generateResponse(prompt) {
        if (!this.context) await this.initialize();
        
        const session = new LlamaChatSession({
            context: this.context
        });
        
        return await session.prompt(prompt);
    }
}
```

### BERT Integration (Python)
```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

class ProductionBERTEngine:
    def __init__(self, config):
        self.model_name = config.get('bert_model', 'sentence-transformers/allenai-specter')
        self.device = config.get('device', 'cpu')
        self.cache_dir = config.get('model_cache_dir')
        self.tokenizer = None
        self.model = None
        
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        ).to(self.device)
        
    async def classify_text(self, text, categories):
        if not self.model:
            self.initialize()
            
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        # Real classification logic here
        return self._compute_category_scores(embeddings, categories)
```

### llama.cpp Integration (Python)
```python
from llama_cpp import Llama

class ProductionLlamaEngine:
    def __init__(self, config):
        self.model_path = config.get('model_path', './models/llama-2-7b-chat.q4_0.gguf')
        self.n_ctx = config.get('context_length', 2048)
        self.n_threads = config.get('threads', 4)
        self.llm = None
        
    def initialize(self):
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )
        
    async def generate_response(self, prompt, max_tokens=512):
        if not self.llm:
            self.initialize()
            
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            echo=False
        )
        
        return output['choices'][0]['text'].strip()
```

## Configuration Requirements

### Environment Variables
```bash
# Required for production deployment
export AI_MODEL_PATH="/path/to/llama/model.gguf"
export BERT_MODEL_NAME="sentence-transformers/allenai-specter"
export MODEL_CACHE_DIR="/opt/ai-models/cache"
export AI_CONTEXT_LENGTH="2048"
export AI_DEVICE="cpu"  # or "cuda" for GPU
export AI_THREADS="4"
export AI_QUANTIZATION="q4_0"
```

## Quality Enforcement

### Prohibited Patterns
```python
# ❌ NEVER DO THIS
def classify_text(text, categories):
    return {"category1": 0.5, "category2": 0.5}  # Mock response

def generate_response(prompt):
    return "This is a mock response"  # Placeholder

async def analyze_content(content):
    # TODO: Implement real AI analysis
    pass
```

### Required Patterns
```python
# ✅ ALWAYS DO THIS
async def classify_text(text, categories):
    engine = await get_production_ai_engine()
    return await engine.classify_text(text, categories)

async def generate_response(prompt):
    engine = await get_production_ai_engine()
    return await engine.generate_response(prompt)

async def analyze_content(content):
    try:
        engine = await get_production_ai_engine()
        return await engine.analyze_content(content)
    except Exception as e:
        # Fallback to smaller model, not mock
        fallback_engine = await get_fallback_ai_engine()
        return await fallback_engine.analyze_content(content)
```

## When Unsure

- Ask for help in integrating llama.cpp properly
- Request guidance on model selection and configuration
- Seek assistance with performance optimization
- **DO NOT** fallback to a mock implementation
- **DO NOT** use simplified placeholder logic
- **DO NOT** skip AI inference requirements


