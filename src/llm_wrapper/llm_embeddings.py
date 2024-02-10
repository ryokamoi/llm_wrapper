from typing import Optional
from pathlib import Path

import openai

from llm_wrapper.cache_utils import read_cached_output, dump_output_to_cache


def llm_embeddings(text: str, model: str="text-embedding-3-small",
                   organization=None, openai_api_key_path: Optional[Path]=Path("../openai_api_key.txt"), overwrite_cache: bool=False, cache_dir: Path=Path("./llm_cache")) -> list[float]:
    """Get the embedding for a piece of text."""
    
    # read cache
    if not overwrite_cache:
        cached_output = read_cached_output(parameters={"model": model}, prompt=text, cache_dir=cache_dir)
        
        if len(cached_output) > 0:
            return cached_output["response"]["data"][0]["embedding"]
    
    # set openai configs
    if organization is not None:
        openai.organization = organization
    
    with open(openai_api_key_path, "r") as f:
        openai.api_key = f.read().strip()
    
    # get response
    response = openai.Embedding.create(
        input=text,
        engine=model
    )
    embeddings = response['data'][0]['embedding']
    
    # dump cache
    if cache_dir is not None:
        dump_output_to_cache(output_dict={"prompt": text, "response": response}, parameters={"model": model}, prompt=text)
    
    return embeddings
