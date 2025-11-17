# %%
#!/usr/bin/env python3
"""
Script to annotate completions using Claude's integrated search.
Identifies which spans are supported, not supported, or have insufficient information.
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
import logging
from dotenv import load_dotenv
from pathlib import Path
import traceback

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse

from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from annotation_pipeline.data_models import AnnotatedSpan

logger = logging.getLogger(__name__)

# %%

ENTITY_ANNOTATION_PROMPT_TEMPLATE = open(os.path.join(Path(__file__).parent, "entity_annotation.prompt")).read().strip()
assert '{instruction}' in ENTITY_ANNOTATION_PROMPT_TEMPLATE \
    and '{completion}' in ENTITY_ANNOTATION_PROMPT_TEMPLATE

# %%

def format_prompt(instruction: str, completion: str, prompt_template: str) -> str:
    """Format the user prompt with the text to analyze"""
    return prompt_template.replace(
        "{instruction}", instruction
    ).replace(
        "{completion}", completion
    )


def assign_span_positions(spans: List[AnnotatedSpan], text: str, min_similarity: float = 0.8) -> List[AnnotatedSpan]:
    """
    Assign positions to spans and convert to the expected format.
    
    Args:
        spans: List of annotated spans from the model
        text: The original text
        
    Returns:
        List of spans with positions added
    """
    results = []
    cur_idx = 0
    used_positions = set()

    for span in spans:

        closest_match, matched_idx = try_matching_span_in_text(
            span.span,
            text,
            cur_idx=cur_idx,
            min_similarity=min_similarity
        )

        if closest_match is None:
            logger.warning(f"Could not locate span {repr(span.span)} in text (total_n_spans: {len(spans)}).\nKeeping it in the dataset but removing the label and index.\n(This is not an error and happens roughly once for every completion)")
            span.label = None
            span.index = None
            results.append(span)
            continue

        if matched_idx is not None and all(pos in used_positions for pos in range(matched_idx, matched_idx+len(closest_match))):
            logger.warning(f"Span {repr(span.span)} matched at same position as already-matched span {repr(text[matched_idx:matched_idx+len(closest_match)])}")
            continue

        if closest_match != span.span:
            logger.info(f"Span {repr(span.span)} matched to {repr(closest_match)}")

        span.index = matched_idx
        span.span = closest_match

        results.append(span)
        used_positions.update(range(matched_idx, matched_idx+len(closest_match)))
        cur_idx = max(cur_idx, matched_idx + len(closest_match))
    
    return results

async def annotate_completion(
    instruction: str,
    completion: str,
    inference_api: InferenceAPI,
    annotation_prompt: str = ENTITY_ANNOTATION_PROMPT_TEMPLATE,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_searches: int = 10,
    model_id: str = "claude-sonnet-4-20250514",
) -> List[AnnotatedSpan]:
    """
    Annotate spans in the provided text completion.
    
    Args:
        instruction: The instruction/prompt that was given to the model
        completion: The completion text to analyze for spans
        inference_api: InferenceAPI instance to use
        temperature: Sampling temperature (0 for deterministic)
        max_tokens: Maximum tokens to generate
        max_searches: Maximum number of web searches to perform
        model: Model to use for annotation
        
    Returns:
        List of annotated spans with their positions
    """
    try:
        user_prompt: str = format_prompt(
            instruction, completion, annotation_prompt
        )

        response: List[LLMResponse] = await inference_api(
            model_id=model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=user_prompt)]),
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_searches
                }
            ]
        )

        response_text: str = response[0].completion

        annotated_spans = parse_and_validate_json(
            response_text, 
            List[AnnotatedSpan],
            allow_partial=True, # try to parse the response even if it's cut off
        )
        
        annotated_spans = assign_span_positions(annotated_spans, completion)
        
        logger.info(f"Successfully annotated {len(annotated_spans)} spans")

        return annotated_spans
        
    except Exception as e:
        logger.error(f"Error during span annotation: {e}")
        raise
