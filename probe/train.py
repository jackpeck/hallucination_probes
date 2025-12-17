"""Training script for hallucination detection probes."""

import argparse
import atexit
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import Subset
from transformers import TrainingArguments

from probe.config import TrainingConfig
from probe.dataset import (
    TokenizedProbingDataset,
    create_probing_dataset,
    tokenized_probing_collate_fn,
)
from probe.trainer import ProbeTrainer
from probe.value_head_probe import setup_probe
from utils.file_utils import load_json, load_yaml, save_json, save_jsonl
from utils.model_utils import load_model_and_tokenizer, print_trainable_parameters
from utils.probe_loader import upload_probe_to_hf


def _normalize_config(obj):
    """Convert config objects to JSON-comparable primitives."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _normalize_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_config(v) for v in obj]
    return obj


def validate_checkpoint_config(
    saved_config_path: Path, current_config: TrainingConfig
) -> None:
    """
    Validate that the saved training config matches the current config exactly.

    Args:
        saved_config_path: Path to the saved training_config.json
        current_config: Current TrainingConfig being used

    Raises:
        AssertionError: If configs differ
    """
    saved_config = _normalize_config(load_json(saved_config_path))
    current_config_dict = _normalize_config(asdict(current_config))

    if saved_config != current_config_dict:
        raise AssertionError(
            "Training config mismatch with saved checkpoint! "
            "Use the same config or delete the checkpoints directory to start fresh."
        )


def main(training_config: TrainingConfig):
    """Main training function."""

    # Load environment variables from .env if present
    load_dotenv()

    if training_config.upload_to_hf:
        assert os.environ.get("HF_WRITE_TOKEN", None) is not None

    wandb.init(
        project=training_config.wandb_project,
        name=training_config.probe_config.probe_id,
    )

    print("Training config:")
    for key, value in asdict(training_config).items():
        print(f"\t{key}: {value}")

    # Load model and tokenizer
    print(f"Loading model: {training_config.probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(training_config.probe_config.model_name)

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if training_config.enable_gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    print(f"Setting up probe: {training_config.probe_config.probe_id}")
    model, probe = setup_probe(model, training_config.probe_config)

    # tell hf Trainer the model is already on the correct devices (using device_map)
    if hasattr(model, "hf_device_map"):
        probe.hf_device_map = model.hf_device_map

    print_trainable_parameters(probe)

    # Load datasets
    print("Loading datasets:")
    train_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.train_dataset_configs
    ]
    eval_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.eval_dataset_configs
    ]

    # Concatenate training datasets
    train_dataset = train_datasets[0]
    for dataset in train_datasets[1:]:
        train_dataset += dataset

    # If requested, shuffle and shave down the training dataset to a fixed number of samples
    if training_config.num_train_samples is not None:
        total = len(train_dataset)
        num = max(0, min(int(training_config.num_train_samples), total))
        if num < total:
            g = torch.Generator()
            g.manual_seed(training_config.seed)
            perm = torch.randperm(total, generator=g).tolist()
            selected_indices = perm[:num]
            train_dataset = Subset(train_dataset, selected_indices)
            print(f"Using a subset of the training dataset: {num}/{total} samples")

    training_args = TrainingArguments(
        output_dir=str(training_config.probe_config.probe_path),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        max_steps=training_config.max_steps,
        num_train_epochs=training_config.num_train_epochs,
        logging_steps=training_config.logging_steps,
        eval_steps=training_config.eval_steps,
        remove_unused_columns=False,
        label_names=["classification_labels", "lm_labels"],
        report_to="wandb",
        run_name=training_config.probe_config.probe_id,
        eval_strategy="steps" if training_config.eval_steps else "no",
        logging_first_step=True,
        logging_strategy="steps",
        max_grad_norm=training_config.max_grad_norm,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        seed=training_config.seed,
    )

    # Add separate learning rates to training_args
    training_args.probe_head_lr = training_config.probe_head_lr
    training_args.lora_lr = training_config.lora_lr

    # Disable checkpoint saving
    # (there's a weird bug that occurs when trying to save during training)
    training_args.set_save(strategy="no")

    trainer = ProbeTrainer(
        probe=probe,
        eval_datasets=eval_datasets,
        cfg=training_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # this is a dummy argument is for the HF base Trainer class
        data_collator=tokenized_probing_collate_fn,
        eval_steps=training_config.eval_steps,
        tokenizer=tokenizer,
    )

    def save_model_callback():
        """Save probe weigths, tokenizer and training config to disk."""
        probe.save(training_config.probe_config.probe_path)
        tokenizer.save_pretrained(training_config.probe_config.probe_path)
        save_json(
            training_config,
            training_config.probe_config.probe_path / "training_config.json",
        )

    # Register save callback for unexpected exits
    atexit.register(save_model_callback)

    # Check for existing checkpoints and auto-resume
    training_config_path = (
        training_config.probe_config.probe_path / "training_config.json"
    )

    resume_from_checkpoint = None
    if training_config_path.exists():
        # Validate config matches before resuming
        validate_checkpoint_config(training_config_path, training_config)

        # Find latest checkpoint
        latest_checkpoint = trainer.get_latest_checkpoint()
        if latest_checkpoint is not None:
            print(f"Found existing checkpoint: {latest_checkpoint}")
            trainer.load_checkpoint(latest_checkpoint)
            resume_from_checkpoint = latest_checkpoint
    else:
        # Save training config at the start (used for validation on resume)
        training_config.probe_config.probe_path.mkdir(parents=True, exist_ok=True)
        save_json(
            training_config,
            training_config_path,
        )
        print(f"Saved training config to {training_config_path}")

    print("Training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the model
    print(f"Saving model to {training_config.probe_config.probe_path}")
    save_model_callback()

    # Final evaluation
    eval_metrics = trainer.evaluate(
        save_roc_curves=training_config.save_roc_curves,
        dump_raw_eval_results=training_config.dump_raw_eval_results,
        verbose=True,
    )

    if training_config.save_evaluation_metrics:
        save_json(
            eval_metrics,
            training_config.probe_config.probe_path / "evaluation_results.json",
        )

    wandb.finish()

    if training_config.upload_to_hf:
        print("Uploading probe to HuggingFace Hub...")
        upload_probe_to_hf(
            repo_id=training_config.probe_config.hf_repo_id,
            probe_id=training_config.probe_config.probe_id,
            token=os.environ.get("HF_WRITE_TOKEN"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a hallucination detection probe"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    # Load config from YAML
    training_config = TrainingConfig(**load_yaml(args.config))

    main(training_config)
