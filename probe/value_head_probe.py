"""ValueHeadProbe: A probe that attaches to a specific layer and predicts hallucinations."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from jaxtyping import Float, Int
from torch import Tensor

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel

from utils.hooks import add_hooks
from utils.model_utils import (
    get_model_layers,
    get_model_hidden_size,
    setup_lora_for_layers
)
from utils.probe_loader import download_probe_from_hf

class ValueHeadProbe(nn.Module):
    """
    A probe that hooks into a specific layer of a language model and uses a linear
    head to predict per-token binary classification (e.g., hallucination detection).
    
    This probe:
    1. Attaches a forward hook to capture hidden states from a specified layer
    2. Applies a linear transformation to predict hallucination scores
    3. Can be used with or without LoRA adapters
    """
    
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, PeftModel],
        layer_idx: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        context_window_size: Optional[int] = 1,
    ):
        """
        Initialize the ValueHeadProbe.
        
        Args:
            model: The language model (with or without LoRA adapters)
            layer_idx: Optional index of the layer to attach the probe to (if not specified we'll infer it from the previously saved probe configuration file)
            path: Optional path to load pre-trained probe weights from
        """
        super().__init__()

        assert layer_idx or path, "Either path or layer index must be provided, otherwise we can't infer the layer where we should hook the value head to."
        
        if path:
            saved_config = json.load(open(path / "probe_config.json"))

            if layer_idx is None:
                layer_idx = saved_config["layer_idx"]
            else:
                # We check that if a layer_idx is provided, it matches the one given in the previously saved config
                assert layer_idx == saved_config["layer_idx"]

        hidden_size = get_model_hidden_size(model)
        model_layers = get_model_layers(model)

        self.model = model
        self.layer_idx = layer_idx
        self.target_module = model_layers[layer_idx]
        self.target_layer_name = self.target_module.__class__.__name__
        self.context_window = []
        
        if not isinstance(model, PeftModel):
            print("WARNING: Model is not a PeftModel. Remember to add LoRA adapters if needed.")
        
        # Initialize the value head (linear layer)
        if path:
            # Load pre-trained weights if path is provided
            self.value_head, _ = ValueHeadProbe.load_head(
                path,
                device=model.device,
                dtype=model.dtype
            )
        else:
            self.value_head = nn.Linear(
                hidden_size * context_window_size, 1, 
                device=model.device, 
                dtype=model.dtype
            )
            print(f"WARNING: Using seed=42 for the initialization of the probe")
            torch.manual_seed(42)
            self._initialize_weights()
        
        # Initialize hook state
        self._hooked_hidden_states: Optional[torch.Tensor] = None
        self._hook_fn = self._get_hook_fn()
    
    def _initialize_weights(self):
        """Initialize the value head weights with small random values."""
        with torch.no_grad():
            self.value_head.weight.data.normal_(mean=0.0, std=0.01)
            if self.value_head.bias is not None:
                self.value_head.bias.data.zero_()
    
    def _get_hook_fn(self):
        """
        Forward hook to capture hidden states from target layer.
        `module_output` is typically [batch_size, seq_len, hidden_dim].
        We do NOT detach it, so gradients can flow back.
        """
        def hook_fn(module, module_input, module_output):
            if isinstance(module_input, tuple) and module_output[0].ndim == 3:
                self._hooked_hidden_states = module_output[0]
            else:
                self._hooked_hidden_states = module_output
        return hook_fn
    
    def forward(
        self,
        input_ids: Int[Tensor, 'batch_size seq_len'],
        attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        labels: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model with probe attached.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for language modeling loss
            **kwargs: Additional arguments passed to the model
        
        Returns:
            Dictionary containing:
                - lm_logits: Language model output logits [batch_size, seq_len, vocab_size]
                - probe_logits: Probe output logits [batch_size, seq_len, 1]
                - lm_loss: Language modeling loss (if labels provided)
        """
        # Reset hooked hidden states
        self._hooked_hidden_states = None
        
        # Set up hooks
        fwd_hooks = [(self.target_module, self._hook_fn)]
        
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            # Forward pass through the model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                **kwargs
            )
        
        # Check that hidden states were captured
        if self._hooked_hidden_states is None:
            raise RuntimeError("Failed to capture hidden states from target layer")
        
        # Build context window
        if len(self.context_window) < self.context_window_size:
            self.context_window.append(self._hooked_hidden_states)
            # Pad with zeros if not enough context
            while len(self.context_window) < self.context_window_size:
                self.context_window.insert(0, torch.zeros_like(self._hooked_hidden_states))
        else:
            self.context_window.pop(0)
            self.context_window.append(self._hooked_hidden_states)
        
        # Concatenate context window hidden states
        context_hidden_states = torch.cat(self.context_window, dim=-1, device=self.value_head.weight.device)

        probe_logits: Float[Tensor, 'batch_size seq_len 1'] = self.value_head(context_hidden_states)

        return {
            "lm_logits": outputs.logits,
            "probe_logits": probe_logits,
            "lm_loss": outputs.loss if hasattr(outputs, 'loss') else None
        }
    
    def save(self, path: Union[str, Path]):
        """
        Save the probe to disk.
        
        Args:
            path: Directory to save the probe to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapters if present
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        
        # Save value head weights
        torch.save(
            self.value_head.state_dict(),
            path / "probe_head.bin"
        )
        
        # Save configuration
        probe_config = {
            "target_layer_name": self.target_module.__class__.__name__,
            "layer_idx": self.layer_idx,
            "hidden_size": self.value_head.in_features
        }
        with open(path / "probe_config.json", 'w') as f:
            json.dump(probe_config, f, indent=4)
        
        print(f"Probe saved to {path}")
    
    @property
    def device(self):
        """Get the device of the model."""
        return self.model.device
    
    @classmethod
    def load_head(
        cls,
        path: Path,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[nn.Module, int]:
        """
        Loads the linear value head from the given path.
        
        Args:
            path: Path to the probe directory
            device: Device to load the probe head on
            dtype: Data type for the probe head
            
        Returns:
            Tuple of (probe_head, layer_idx)
        """
        with open(path / "probe_config.json") as f:
            probe_config = json.load(f)

        hidden_size = probe_config['hidden_size']
        probe_layer_idx = probe_config['layer_idx']

        probe_head = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
        
        state_dict = torch.load(
            path / "probe_head.bin",
            map_location=device,
            weights_only=True
        )
        probe_head.load_state_dict(state_dict)
        
        return probe_head, probe_layer_idx


def setup_probe(
    model: PreTrainedModel,
    probe_config: 'ProbeConfig'
) -> Tuple[PreTrainedModel, ValueHeadProbe]:
    """
    Set up a probe with the given configuration.
    
    This handles downloading from HF if needed, setting up LoRA adapters,
    and creating the ValueHeadProbe instance.
    
    Args:
        model: The base language model
        probe_config: Probe configuration
        
    Returns:
        Tuple of (model with LoRA if applicable, probe instance)
    """
    
    # We freeze all parameters of the base model
    # otherwise the optimizer will do a full-finetuning
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Download from HF if needed
    if probe_config.load_from == 'hf' and not probe_config.probe_path.exists():
        download_probe_from_hf(
            repo_id=probe_config.hf_repo_id,
            probe_id=probe_config.probe_id
        )

    if probe_config.load_from in ['hf', 'disk']:
        # Set up the LoRAs (if applicable)
        if (probe_config.probe_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, probe_config.probe_path)

        # Load existing probe (layer_idx will be inferred from previously saved config)
        probe = ValueHeadProbe(model, path=probe_config.probe_path)

    else:
        # Initialize the probe from scratch
        if probe_config.lora_layers:
            print(f"Initializing new LoRA adapters for layers {probe_config.lora_layers}")
            model = setup_lora_for_layers(
                model,
                probe_config.lora_layers,
                lora_r=probe_config.lora_r,
                lora_alpha=probe_config.lora_alpha,
                lora_dropout=probe_config.lora_dropout,
            )
        
        # Initialize new probe with specified layer

        probe = ValueHeadProbe(
            model, 
            layer_idx=probe_config.layer,
            context_window_size=probe_config.context_window_size
        )
  
    return model, probe
