# LLM Entity Matching Evaluation Framework

A modular framework for evaluating small language models on entity matching tasks, comparing their performance against baseline approaches.

## 📁 Project Structure

```
.
├── config.py                    # Configuration management
├── llm_client.py               # LLM API client
├── evaluator.py                # Evaluation logic
├── reporter.py                 # Report generation
├── pipeline.py                 # Main evaluation pipeline
├── data_preparation.py         # Data loading utilities
├── main.py                     # Main execution script
├── test_config.py              # Test configuration loading
├── test_data_preparation.py    # Test data loading
├── test_single_evaluation.py   # Test single model
├── models_config.json          # Model configurations
├── .env                        # API keys (create this)
└── data/
    └── data.parquet           # Your evaluation data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn python-dotenv openai anthropic google-generativeai
```

### 2. Set Up Configuration

Create `.env` file with your API keys:
```bash
OPENAI_API_KEY=sk-proj-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-key
```

Create `models_config.json` (or use the provided template):
```json
{
  "models": [
    {
      "name": "GPT-4o-mini",
      "provider": "openai",
      "model_id": "gpt-4o-mini",
      "api_key": "",
      "api_key_env": "OPENAI_API_KEY",
      "cost_per_1m_input_tokens": 0.15,
      "cost_per_1m_output_tokens": 0.60,
      "max_tokens": 100
    }
  ]
}
```

### 3. Test Each Component

```bash
# Test configuration loading
python test_config.py

# Test data loading (adjust path in script)
python test_data_preparation.py

# Test single model evaluation (makes real API calls!)
python test_single_evaluation.py
```

### 4. Run Full Evaluation

```bash
# Edit main.py to set your data path, then run:
python main.py
```

## 📦 Module Details

### `config.py`
- `ModelConfig`: Dataclass for model configuration
- `load_models_config()`: Load models from JSON file
- `PROMPT_TEMPLATE`: Default prompt for entity matching

### `llm_client.py`
- `LLMClient`: Unified interface for OpenAI, Anthropic, Google
- Handles API calls and token counting

### `evaluator.py`
- `EvaluationResult`: Dataclass for evaluation results
- `evaluate_llm()`: Run evaluation on a single model
- `parse_llm_response()`: Parse MATCH/NO_MATCH responses

### `reporter.py`
- `create_comparison_report()`: Generate comparison tables
- Identifies best models by different criteria
- Saves reports to file

### `pipeline.py`
- `EntityMatchingEvaluator`: Main evaluation orchestrator
- `load_models()`: Load and validate model configs
- `run_evaluation()`: Run full evaluation pipeline

### `data_preparation.py`
- `extract_name()`, `extract_address()`, `extract_category()`: JSON extractors
- `prepare_evaluation_data()`: Transform raw data
- `load_and_prepare_data()`: Load from parquet and prepare

## 🧪 Testing Strategy

### Level 1: Configuration
```bash
python test_config.py
```
✅ Tests: Config file loading, API key resolution, model validation

### Level 2: Data Loading
```bash
python test_data_preparation.py
```
✅ Tests: Parquet loading, JSON extraction, data transformation

### Level 3: Single Model
```bash
python test_single_evaluation.py
```
✅ Tests: API calls, evaluation metrics, cost calculation
⚠️ Makes real API calls (small cost)

### Level 4: Full Pipeline
```bash
python main.py
```
✅ Tests: Full multi-model evaluation
⚠️ Makes many API calls (evaluate costs first)

## 📊 Expected Data Format

Your parquet file should have these columns:
- `names`: JSON with `{"primary": "Name A"}`
- `base_names`: JSON with `{"primary": "Name B"}`
- `addresses`: JSON with `[{"freeform": "123 Main St"}]` (optional)
- `base_addresses`: JSON with `[{"freeform": "123 Main St"}]`
- `categories`: JSON with `{"primary": "restaurant"}`
- `base_categories`: JSON with `{"primary": "restaurant"}`
- `label`: Ground truth (1 = match, 0 = no match)

## 📈 Output Files

After running evaluation:
- `evaluation_log.txt`: Detailed logs for each model
- `evaluation_report.txt`: Comparison table and recommendations
- `llm_evaluation_results.json`: Structured results data

## 💰 Cost Control

- Set `MAX_SAMPLES` in `main.py` to limit evaluation size
- Use `max_samples` parameter in `run_evaluation()`
- Test with `test_single_evaluation.py` first (only 5 samples)

## 🔧 Customization

### Add New Model Provider
Edit `llm_client.py` to add support for new providers:
```python
elif self.config.provider == "your_provider":
    # Add your provider logic
```

### Custom Prompt
Edit `PROMPT_TEMPLATE` in `config.py` or pass custom prompt to `evaluate_llm()`

### Different Metrics
Edit `evaluator.py` to add custom metrics to `EvaluationResult`

## ⚠️ Important Notes

1. **API Keys**: Never commit `.env` or API keys to version control
2. **Costs**: Monitor API usage - costs can add up quickly
3. **Rate Limits**: Some providers have rate limits on API calls
4. **Testing**: Always test with small samples first

## 📝 Example Workflow

```python
from pipeline import EntityMatchingEvaluator
from data_preparation import load_and_prepare_data

# Load data
data = load_and_prepare_data("data/data.parquet", sample_size=100)

# Initialize evaluator
evaluator = EntityMatchingEvaluator()
evaluator.load_models()

# Run evaluation
results = evaluator.run_evaluation(
    test_data=data,
    max_samples=100,
    baseline_result={'f1_score': 0.75, 'precision': 0.73}
)
```

## 🐛 Troubleshooting

**"No API key found"**: Check `.env` file and `api_key_env` in config

**"FileNotFoundError"**: Verify data path and config file exist

**"Provider error"**: Ensure provider libraries are installed (openai, anthropic, google-generativeai)

**Rate limit errors**: Add delays between API calls or reduce sample size