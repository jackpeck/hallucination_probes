# Real-Time Detection of Hallucinated Entities in Long-Form Generation

This is the codebase corresponding to the paper 'Real-Time Detection of Hallucinated Entities in Long-Form Generation':
- Paper link: [arxiv.org/abs/2509.03531](https://arxiv.org/abs/2509.03531)
- Project website: [hallucination-probes.com](https://www.hallucination-probes.com/)

## Datasets

All long-form datasets are provided as a [HuggingFace collection](https://huggingface.co/collections/obalcells/hallucination-probes-68bb658a4795f9294a73b991). This includes:
- Token-level annotations of long-form generations:
  - [LongFact annotations](https://huggingface.co/datasets/obalcells/longfact-annotations)
  - [LongFact++ annotations](https://huggingface.co/datasets/obalcells/longfact-augmented-annotations)
  - [HealthBench annotations](https://huggingface.co/datasets/obalcells/healthbench-annotations)
- Prompts used to elicit long-form generations:
  - [LongFact++ prompts](https://huggingface.co/datasets/obalcells/longfact-augmented-prompts)

## Pretrained Probes

Pretrained hallucination detection probes for various LLMs are available at: [obalcells/hallucination-probes](https://huggingface.co/obalcells/hallucination-probes)

We provide three types of probes:
- **Linear probes** (`*_linear`): Simple linear classifiers trained on model hidden states
- **LoRA probes with KL regularization** (`*_lora_lambda_kl_0_05`): LoRA adapters with KL divergence regularization (λ=0.05) for minimal impact on generation quality
- **LoRA probes with LM regularization** (`*_lora_lambda_lm_0_01`): LoRA adapters with cross-entropy loss regularization (λ=0.01)

Supported models include:
- Llama 3.3 70B
- Llama 3.1 8B
- Gemma 2 9B
- Mistral Small 24B
- Qwen 2.5 7B

## Code

### Setup

To set environment variables, copy `env.example` to `.env` and fill in values.

Run the following to get set up using `uv`:

```bash
# Install Python 3.10 and create env
uv python install 3.10
uv venv --python 3.10

# Sync dependencies
uv sync
```

### Training a probe

Edit `configs/train_config.yaml` as needed (model, datasets, LoRA layers, learning rates). Then run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m probe.train --config configs/train_config.yaml
```

Outputs (by default) are saved under `value_head_probes/{probe_id}`. To upload to Hugging Face, set `upload_to_hf: true` in the config and be sure to set `HF_WRITE_TOKEN` in your `.env` file.

### Running the annotation pipeline

This pipeline uses a frontier LLM with web search to label entities and align token-level spans. Environment variables required:

```bash
export ANTHROPIC_API_KEY=...   # for annotation
export HF_WRITE_TOKEN=...      # to push to HF datasets
```

Run (see `annotation_pipeline/README.md` and `run.py` for full arguments):

```bash
uv run python -m annotation_pipeline.run \
  --model_id "ANTHROPIC_MODEL_ID" \
  --hf_dataset_name "ORG/DATASET" \
  --hf_dataset_subset "SUBSET" \
  --hf_dataset_split "SPLIT" \
  --output_hf_dataset_name "ORG/OUTPUT_DATASET" \
  --output_hf_dataset_subset "SUBSET" \
  --parallel true \
  --max_concurrent_tasks N_CONNCURRENT
```

As a sample command, you can run:

```bash
uv run python -m annotation_pipeline.run \
  --model_id "claude-sonnet-4-20250514" \
  --hf_dataset_name "obalcells/labeled-entity-facts" \
  --hf_dataset_subset "annotated_Meta-Llama-3.1-8B-Instruct" \
  --hf_dataset_split "test" \
  --output_hf_dataset_name "andyrdt/labeled-entity-facts-test" \
  --output_hf_dataset_subset "annotated_Meta-Llama-3.1-8B-Instruct" \
  --parallel true \
  --max_concurrent_tasks 10
```

### Demo UI

The demo provides a real-time visualization of hallucination detection during text generation. It consists of:

- **Backend**: `demo/modal_backend.py` - A Modal app with vLLM that loads the target model and applies probe heads (and optional LoRA) to compute token-level probabilities during generation.
- **Frontend**: `demo/probe_interface.py` - A Streamlit interface that connects to the Modal backend and visualizes token-level confidence scores.

#### Prerequisites

1. **Set up Modal**:
   - Create a Modal account at [https://modal.com/signup](https://modal.com/signup) (as of August 2025, they provide $30 in free credits for new accounts)
   - Install Modal: `pip install modal`
   - Run `modal setup` to authenticate

2. **Environment variables** (add to `.env`):

   ```bash
    HF_TOKEN=your_huggingface_token_id
   ```

3. **Select a probe**: The Modal backend requires you to specify which probe to load. Available probe names include:

   For Llama 3.1 8B:
   - `llama3_1_8b_lora_lambda_kl_0_05` - LoRA probe with high KL regularization (recommended)
   - `llama3_1_8b_linear` - Linear probe
   - `llama3_1_8b_lora_lambda_lm_0_01` - LoRA probe with LM regularization

   For Llama 3.3 70B:
   - `llama3_3_70b_lora_lambda_kl_0_05` - LoRA probe with high KL regularization (recommended)
   - `llama3_3_70b_linear` - Linear probe
   - `llama3_3_70b_lora_lambda_lm_0_01` - LoRA probe with LM regularization

   **Recommendation**: Use the `*_lora_lambda_kl_0_05` probes for the best results and smallest impact on generation quality.

#### Running the Demo

Both the Modal backend and Streamlit frontend must be run from inside the `demo/` directory:

```bash
# Navigate to the demo directory
cd demo

# Deploy the Modal backend
modal deploy modal_backend.py

# Run the Streamlit frontend (also from demo/)
streamlit run probe_interface.py
```

Open your browser to use the interface. The interface will connect to your deployed Modal backend and allow you to input prompts, generate text, and see real-time hallucination detection with color-coded tokens based on the probe's confidence scores.

## Citation

```bibtex
@misc{obeso2025realtimedetectionhallucinatedentities,
      title={Real-Time Detection of Hallucinated Entities in Long-Form Generation}, 
      author={Oscar Obeso and Andy Arditi and Javier Ferrando and Joshua Freeman and Cameron Holmes and Neel Nanda},
      year={2025},
      eprint={2509.03531},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03531}, 
}
```

