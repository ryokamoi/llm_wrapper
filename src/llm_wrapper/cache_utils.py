from typing import Union
import json
from pathlib import Path
import hashlib


def text2hash(string: str) -> str:
    hash_object = hashlib.sha512(string.encode('utf-8'))
    hex_dig = hash_object.hexdigest()

    return hex_dig


def get_cache_path(cache_dir: Path, parameters: dict, prompt: str) -> Path:
    return cache_dir / f"{text2hash(str(parameters) + prompt)}.json"


def read_cached_output(parameters: dict, prompt: str, cache_dir: Path = Path("./llm_cache")) -> Union[dict, list[dict]]:
    """Read cached output from cache_dir. When temperature is zero, the output is dict. Otherwise, the output is list[dict]."""
    
    cache_path = get_cache_path(cache_dir, parameters, prompt)
    if cache_path.exists():
        print(f"read cache from {cache_path}")
        with open(cache_path, "r") as f:
            output_dict = json.load(f)
        return output_dict
    
    return {}


def dump_output_to_cache(output_dict: dict, parameters: dict, prompt: str, cache_dir: Path = Path("./llm_cache")):
    cache_path = get_cache_path(cache_dir, parameters, prompt)
    
    cache_dir.mkdir(exist_ok=True, parents=True)
    with open(cache_path, "w") as f:
        json.dump(output_dict, f, indent=4)
