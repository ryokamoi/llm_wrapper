from typing import Any
from pathlib import Path

from llm_wrapper import llm_api, preprocess_parameters
from llm_wrapper.cache_utils import read_cached_output, dump_output_to_cache


def self_consistency(model_name: str, prompt: str, sample_num: int, updated_parameters: dict={}, overwrite_all_cache: bool=False,
                     loaded_model: Any=None) -> list[dict]:
    """Get k samples from LLM. You may set the parameters by using parameter_update. Output format is a list of {"prompt": str, "response": str}.
    Cache will be stored in cache_dir. """
    
    if updated_parameters["temperature"] == 0 or "temperature" not in updated_parameters.keys():
        raise ValueError("temperature should be greater than 0 for self-consistency.")
    
    sample_list: list[dict] = []

    processed_parameters = preprocess_parameters(model_name, updated_parameters)
    processed_parameters["model"] = model_name  # fix this part later
    
    # read cached output
    cached_output: list[dict] = []
    if not overwrite_all_cache:
        cached_output = read_cached_output(parameters=processed_parameters, prompt=prompt,
                                           cache_dir=Path("./llm_cache"))
        if len(cached_output) > 0:  # when len(cached_output) == 0, the output is not cached and an empty dict is returned
            sample_list = cached_output[:sample_num]
    
    # add new outputs if cached output do not include enough samples
    while len(sample_list) < sample_num:
        output_dict = llm_api(model_name=model_name, prompt=prompt, overwrite_cache=True, updated_parameters=updated_parameters,
                              loaded_model=loaded_model)
        sample_list.append(output_dict)
    
    # dump cache if there are new samples
    if len(cached_output) < len(sample_list):
        dump_output_to_cache(output_dict=sample_list, parameters=processed_parameters, prompt=prompt,
                             cache_dir=Path("./llm_cache"))
    
    return sample_list
