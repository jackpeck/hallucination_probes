#!/usr/bin/env python3
"""
Script to run the annotation pipeline on any HuggingFace dataset.

This pipeline can process any dataset as long as it contains a 'conversation' field.
All other fields in the dataset will be preserved automatically.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import simple_parsing
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm_asyncio
import hashlib
import traceback

from safetytooling.utils.experiment_utils import ExperimentConfigBase
from safetytooling.apis import InferenceAPI

from utils.file_utils import load_jsonl, save_jsonl
from utils.parsing import validate_dicts_to_pydantic
from annotation_pipeline.annotate import annotate_completion
from annotation_pipeline.data_models import DatasetItem, AnnotatedSpan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Ensure required environment variables are set
assert os.environ.get('ANTHROPIC_API_KEY', None) is not None, "ANTHROPIC_API_KEY is not set"
assert os.environ.get('HF_WRITE_TOKEN', None) is not None, "HF_WRITE_TOKEN is not set (needed for pushing to HF hub)"

LOCAL_RESULTS_DIR = Path(__file__).parent.parent / "annotation_pipeline_results"


@dataclass
class PipelineConfig(ExperimentConfigBase):
    """Configuration for running the annotation pipeline.
    
    Can be used with any HuggingFace dataset that contains a 'conversation' field.
    All other fields in the dataset will be automatically preserved.
    """
    
    # Model and API settings
    model_id: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_searches: int = 15
    max_tokens: int = 8192
    
    # Input dataset settings
    hf_dataset_name: str = "obalcells/generations"
    hf_dataset_subset: str = "healthbench_subset"
    hf_dataset_split: str = "test"
    
    # Output settings
    output_hf_dataset_name: str = "obalcells/labeled-entity-facts"
    output_hf_dataset_subset: str = "clean_code_test"
    output_hf_dataset_split: str = "test"
    output_dir: Path = LOCAL_RESULTS_DIR
    
    # Processing settings
    parallel: bool = True
    verbose: bool = True
    max_concurrent_tasks: int = 25 # Limit concurrent API calls to avoid rate limiting
    push_intermediate_every: int = 1_000  # Push to HF every N processed items (0 to disable)
    
    # Args needed by the ExperimentConfigBase base class
    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"
    save_path: Optional[Path] = None
    log_to_file: bool = False
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self.run_id = f"{self.hf_dataset_name.replace('/', '_')}_{self.hf_dataset_subset}_{self.hf_dataset_split}"

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if self.save_path is None:
            self.save_path = self.output_dir / f"{self.run_id}.jsonl"


def get_item_key(item: DatasetItem) -> str:
    """Generate a unique key for an item based on its completion text"""
    content_hash = hashlib.md5(item.conversation[-1]['content'].encode()).hexdigest()
    return content_hash


def is_item_processed(item: DatasetItem, processed_keys: set) -> bool:
    """Check if an item has already been processed"""
    return get_item_key(item) in processed_keys


def load_processed_items_from_disk(file_path: Path) -> List[DatasetItem]:
    """
    Load already-processed items from a JSONL file and validate them as DatasetItem objects.
    
    Args:
        file_path: Path to the JSONL file containing processed items
        
    Returns:
        List of validated DatasetItem objects
    """
    if not file_path.exists():
        return []
    
    try:
        raw_items = load_jsonl(file_path)
    except Exception as e:
        logger.warning(f"Could not load from file {file_path}: {e}")
        return []
    
    # Use the generic validation function with skip_invalid=True
    validated_items = validate_dicts_to_pydantic(raw_items, DatasetItem, skip_invalid=True)
    
    # Log if some items failed validation
    if len(validated_items) < len(raw_items):
        logger.warning(f"Skipped {len(raw_items) - len(validated_items)} invalid items from {file_path}")
    
    return validated_items


def load_processed_item_keys(cfg: PipelineConfig) -> set:
    """Load already processed item keys from both local file and HuggingFace"""
    processed_keys = set()

    # Load from local file
    local_items = load_processed_items_from_disk(cfg.save_path)
    for item in local_items:
        processed_keys.add(get_item_key(item))
    
    if local_items:
        logger.info(f"Loaded {len(local_items)} already processed items from local file {cfg.save_path}")
    
    # Load from HuggingFace
    try:
        logger.info(f"Loading processed items from HuggingFace: {cfg.output_hf_dataset_name}/{cfg.output_hf_dataset_subset}/{cfg.output_hf_dataset_split}")
        hf_dataset = load_dataset(
            cfg.output_hf_dataset_name, 
            cfg.output_hf_dataset_subset, 
            split=cfg.output_hf_dataset_split,
        )
        
        # Convert HF dataset to list of dicts and validate
        hf_items_raw = list(hf_dataset)
        hf_items_validated = validate_dicts_to_pydantic(hf_items_raw, DatasetItem, skip_invalid=True)
        
        for item in hf_items_validated:
            processed_keys.add(get_item_key(item))
        
        logger.info(f"Loaded {len(hf_items_validated)} already processed items from HuggingFace")
        
    except Exception as e:
        logger.warning(f"Could not load from HuggingFace (dataset may not exist yet): {e}")
    
    logger.info(f"Total unique processed items: {len(processed_keys)}")
    return processed_keys


def load_items_to_process(cfg: PipelineConfig) -> List[DatasetItem]:
    """Load dataset items that need to be processed"""
    # Load the dataset
    dataset = load_dataset(
        cfg.hf_dataset_name,
        cfg.hf_dataset_subset,
        split=cfg.hf_dataset_split
    )

    # Check that the dataset has the required columns
    required_columns = ["conversation"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset is missing required column: {col}. Available columns: {dataset.column_names}")

    logger.info(f"Dataset validation successful: required columns {required_columns} are present")
    logger.info(f"Dataset has {len(dataset)} items with columns: {dataset.column_names}")
    
    # Convert to list of dictionaries (keep all columns)
    dataset_items_raw = list(dataset)
    logger.info(f"Loaded dataset: {len(dataset_items_raw)} items")
    
    # Convert to DatasetItem objects using the generic validation function
    dataset_items = validate_dicts_to_pydantic(dataset_items_raw, DatasetItem, skip_invalid=True)
    
    # Log if some items failed validation
    if len(dataset_items) < len(dataset_items_raw):
        logger.warning(f"Skipped {len(dataset_items_raw) - len(dataset_items)} invalid items from dataset")
    
    logger.info(f"Successfully validated {len(dataset_items)} items")
    
    # Load already processed items
    processed_keys = load_processed_item_keys(cfg)
    
    # Filter out already processed items
    items_to_process = [item for item in dataset_items if not is_item_processed(item, processed_keys)]
    logger.info(f"Filtered out {len(dataset_items) - len(items_to_process)} already processed items")
    logger.info(f"Items to process: {len(items_to_process)}")
    
    return items_to_process


def sync_to_huggingface(cfg: PipelineConfig) -> None:
    """Sync all processed items from local .jsonl file to HuggingFace Hub, deduplicating with existing items."""
    
    # Always load all processed items from local file to ensure nothing is missed
    all_local_items = load_processed_items_from_disk(cfg.save_path)
    
    if all_local_items:
        logger.info(f"Loaded {len(all_local_items)} items from local file {cfg.save_path}")
    
    if not all_local_items:
        logger.info("No items to push to HuggingFace")
        return
    
    try:
        # Load existing items from HuggingFace
        existing_items = {}
        try:
            logger.warning(f"Loading existing items from HuggingFace for deduplication")
            hf_dataset = load_dataset(
                cfg.output_hf_dataset_name, 
                cfg.output_hf_dataset_subset, 
                split=cfg.output_hf_dataset_split
            )
            
            # Convert HF dataset to list of dicts and validate
            hf_items_raw = list(hf_dataset)
            hf_items_validated = validate_dicts_to_pydantic(hf_items_raw, DatasetItem, skip_invalid=True)
            
            for item in hf_items_validated:
                existing_items[get_item_key(item)] = item
            
            logger.info(f"Loaded {len(existing_items)} existing items from HuggingFace")
        except Exception as e:
            logger.error(f"Could not load pre-existing items from HuggingFace (dataset may not exist yet): {e}")
        
        # Combine local items with existing HF items, deduplicating
        combined_items = dict(existing_items)

        for item in all_local_items:
            combined_items[get_item_key(item)] = item  # overwrite HF with local
        
        all_items = list(combined_items.values())
        
        logger.info(f"Pushing {len(all_items)} total items, {len(existing_items)} existing) to {cfg.output_hf_dataset_name}/{cfg.output_hf_dataset_subset}/{cfg.output_hf_dataset_split}")
        
        # Convert Pydantic models to dictionaries for HF dataset
        processed_dicts_for_hf = [item.model_dump() for item in all_items]
        hf_dataset = Dataset.from_list(processed_dicts_for_hf)
        hf_dataset.push_to_hub(
            cfg.output_hf_dataset_name,
            cfg.output_hf_dataset_subset,
            split=cfg.output_hf_dataset_split,
            token=os.environ.get("HF_WRITE_TOKEN")
        )
        
        logger.info(f"Successfully pushed {len(all_items)} items to {cfg.output_hf_dataset_name}/{cfg.output_hf_dataset_subset}/{cfg.output_hf_dataset_split}")
        
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace: {e}")
        logger.error(traceback.format_exc())


async def process_dataset_item(
    item: DatasetItem,
    cfg: PipelineConfig,
    inference_api: InferenceAPI
) -> Optional[DatasetItem]:
    """
    Process a single dataset item.
    
    Args:
        item: A DatasetItem containing the dataset item
        cfg: The experiment configuration
        inference_api: The inference API to use
        
    Returns:
        The processed DatasetItem with added 'annotations' field, or None if processing failed
    """
    if item is None:
        return None

    if not item.conversation or len(item.conversation) < 2:
        logger.warning("Skipping item with <2 conversation turns")
        return None

    try:
        # The last message is the completion, the second to last is the instruction
        # (in all conversations in our dataset we simply have two messages, but we could have more
        # if we wanted to have a multi-turn conversation dataset)
        instruction: str = item.conversation[-2]['content']
        completion: str = item.conversation[-1]['content']

        annotations: List[AnnotatedSpan] = await annotate_completion(
            instruction=instruction,
            completion=completion,
            inference_api=inference_api,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_searches=cfg.max_searches,
            model_id=cfg.model_id,
        )

        item.annotations = annotations

        if annotations is None:
            return None
        
        # Save intermediate result as dictionary
        save_jsonl(item.model_dump(), cfg.save_path, append=True)
            
        if cfg.verbose:
            labels: List[str] = [e.label for e in item.annotations or []]
            label_counts = {label: len([l for l in labels if l == label]) for label in set(labels)}
            logger.info(f"Successfully processed item with label counts: {label_counts}, total annotations: {len(labels)}")
            
        return item
            
    except Exception as e:
        logger.error(f"Error processing item: {e}")
        logger.error(traceback.format_exc())
        return None


async def main(cfg: PipelineConfig):
    """Main function to run the annotation pipeline"""
    cfg.setup_experiment()

    os.makedirs(cfg.output_dir, exist_ok=True)
    
    items_to_process = load_items_to_process(cfg)
    
    if not items_to_process:
        logger.info("No items to process. All items have already been processed.")
        return
    
    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
        openai_num_threads=cfg.max_concurrent_tasks,
        deepseek_num_threads=cfg.max_concurrent_tasks,
    )
    
    if not cfg.parallel:
        # Sequential processing
        logger.info("Processing items sequentially...")
        processed_count = 0
        
        for i, item in enumerate(items_to_process):
            logger.info(f"Processing item {i+1}/{len(items_to_process)}")
            processed_item = await process_dataset_item(item, cfg, inference_api)
            if processed_item:
                logger.info(f"Successfully processed item {i+1}")
                processed_count += 1
                
                # Push intermediate results if enabled
                if (cfg.push_intermediate_every > 0 and 
                    processed_count % cfg.push_intermediate_every == 0):
                    
                    logger.info(f"Pushing intermediate results after {processed_count} items...")
                    
                    sync_to_huggingface(cfg)
                    
            else:
                logger.error(f"Error processing item {i+1}")
    else:
        # Parallel processing (API class handles concurrency control)
        logger.info(f"Processing {len(items_to_process)} items in parallel")
        
        if cfg.push_intermediate_every > 0:
            logger.info(f"Will push intermediate results every {cfg.push_intermediate_every} items")
        
        # Process in batches if intermediate pushing is enabled
        if cfg.push_intermediate_every > 0:
            batch_size = cfg.push_intermediate_every
            processed_count = 0
            
            for batch_start in range(0, len(items_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(items_to_process))
                batch_items = items_to_process[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}: items {batch_start+1}-{batch_end}")
                
                # Process batch in parallel
                tasks = [process_dataset_item(item, cfg, inference_api) for item in batch_items]
                results = await tqdm_asyncio.gather(*tasks, desc=f"Processing batch {batch_start}")
                
                # Count successful results in this batch
                batch_successful = sum(1 for r in results if r is not None)
                processed_count += batch_successful
                logger.info(f"Batch completed: {batch_successful}/{len(batch_items)} items successful")
                
                # Push intermediate results after this batch
                logger.info(f"Pushing intermediate results after {processed_count} total items...")
                
                sync_to_huggingface(cfg)
        else:
            # Process all items at once (original behavior)
            tasks = [process_dataset_item(item, cfg, inference_api) for item in items_to_process]
            results = await tqdm_asyncio.gather(*tasks, desc="Processing items")
            
            # Count successful results
            successful = sum(1 for r in results if r is not None)
            logger.info(f"Successfully processed {successful}/{len(items_to_process)} items")
    
    # Sync final results to HuggingFace (will deduplicate with existing items)
    sync_to_huggingface(cfg)

    print("=" * 60)
    print("ANNOTATION RUN COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(PipelineConfig, dest="experiment_config")
    
    args = parser.parse_args()
    experiment_config: PipelineConfig = args.experiment_config
    
    asyncio.run(main(experiment_config)) 