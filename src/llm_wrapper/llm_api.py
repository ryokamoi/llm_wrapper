from pathlib import Path
import time
from typing import Optional, TypedDict, Any

import easy_io

from llm_wrapper.utils import is_openai_model, is_gemini, is_claude, is_open_model
from llm_wrapper.cache_utils import read_cached_output, dump_output_to_cache
from llm_wrapper.open_models import OpenModel


gpt_parameters: dict = {
    "model": "",
    "temperature": 0.,
    "max_tokens": None,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

palm_parameters: dict = {
    "model": '',
    "temperature": 0,
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "top_p": None,
    "top_k": None,
}

cohere_parameters: dict = {
    "model": '',
    "max_tokens": 512,
    "temperature": 0,
    "k": 0,
    "stop_sequences": [],
    "return_likelihoods": 'NONE'
}


def preprocess_parameters(model_name: str, updated_parameters: dict) -> dict:
    parameters = {}
    if not is_openai_model(model_name):
        if is_open_model(model_name):
            parameters = {"model": model_name}
            
            # do_sample = False case
            if "do_sample" in updated_parameters.keys():
                if not updated_parameters["do_sample"]:
                    updated_parameters["temperature"] = 0

            if "temperature" in updated_parameters:
                if updated_parameters["temperature"] == 0:
                    updated_parameters["do_sample"] = False
                    updated_parameters["temperature"] = None
                    updated_parameters["top_p"] = None
                    updated_parameters["top_k"] = None
                else:
                    updated_parameters["do_sample"] = True
        elif is_gemini(model_name):
            if "models/" not in model_name:
                model_name = f"models/{model_name}"
            
            if is_gemini == "palm":
                parameters = dict(palm_parameters, model=model_name)
            else:  # gemini
                parameters = {"model": model_name}
        elif is_claude(model_name):
            parameters = {"model": model_name}
        elif "command" in model_name:
            parameters = dict(cohere_parameters, model=model_name)
        else:
            raise ValueError(f"{model_name} is an invalid value for model_name argument.")
        
        parameters = dict(parameters, **updated_parameters)
    
    return parameters


def loop_process(error_message: str, model_name: str, prompt: str, loop_count: int, loop_limit: int, sleep_time: int=5):
    print(error_message)
    if loop_count == loop_limit -1:
        raise Exception(f"Received {loop_limit} Response Error for the same input from {model_name}. Please try again later.\nPrompt:\n{prompt}")
    print(f"Wait for {sleep_time} seconds.")
    time.sleep(sleep_time)


class LlmApiOutput(TypedDict):
    prompt: str
    response: str


def llm_api(model_name: str, prompt: str, updated_parameters: dict={},
            loaded_model: Any=None,
            overwrite_cache: bool=False, cache_dir: Path=Path("./llm_cache"),
            sleep_time: int=-1,
            openai_organization: Optional[str]=None) -> LlmApiOutput:
    """Call LLM APIs. You may set the parameters by using parameter_update. Output format is {"prompt": str, "response": str}. Cache will be stored in cache_dir.
    
    Args:
        sleep_time (int, optional): Sleep time in seconds. If the value is negative, the default value for each model is used. Defaults to -1."""

    parameters = preprocess_parameters(model_name=model_name, updated_parameters=updated_parameters)    
    
    # read cache
    cached_output = {}
    if not overwrite_cache and not is_openai_model(model_name):
        # cache for openai models will be handled by openai_api_wrapper
        
        # for other models:
        cached_output = read_cached_output(parameters=parameters, prompt=prompt, cache_dir=cache_dir)
        
        # if cache is found, return the cache
        if len(cached_output) > 0:
            # Palm can output None. If the cache includes None, the cache will be ignored.
            if cached_output["response"] is None:
                cached_output = {}
            
            # if the above conditions are not applicable, the cache_output includes a propoer cache
            if len(cached_output) > 0:
                return cached_output
            # otherwise, the cache is ignored and the code will continue to call the api
    
    # llm api
    if is_openai_model(model_name):  # openai models
        from llm_wrapper.openai_models import openai_text_api, get_chat_parameters
        
        # update parameters
        if "model" in updated_parameters:
            assert updated_parameters["model"] == model_name
        else:
            gpt_parameters["model"] = model_name
        updated_gpt_parameters = dict(gpt_parameters, **updated_parameters)
        
        from functools import partial
        openai_text_api_partially_filled = partial(openai_text_api,
                                                   cache_dir=cache_dir, overwrite_cache=overwrite_cache, organization=openai_organization,
                                                   sleep_time=0 if sleep_time < 0 else sleep_time)
        
        # call api
        output = openai_text_api_partially_filled(mode="chat", parameters=get_chat_parameters(prompt=prompt, parameters=updated_gpt_parameters))
        response = output["response"]["choices"][0]["message"]["content"]
        # if is_model_for_chat(model_name):  # chat models like gpt-4
        #     output = openai_text_api_partially_filled(mode="chat", parameters=get_chat_parameters(prompt=prompt, parameters=updated_gpt_parameters))
        #     response = output["response"]["choices"][0]["message"]["content"]
        # else:  # completion models like text-davinci-003
        #     output = openai_text_api_partially_filled(mode="complete", parameters=dict(updated_gpt_parameters, prompt=prompt))
        #     response = output["response"]["choices"][0]["text"]
    else:
        if is_open_model(model_name):
            assert loaded_model is not None
            
            loaded_model: OpenModel = loaded_model
            response = loaded_model(prompt, parameters=parameters)
        
        # LLM APIs can return errors even when the input is valid (e.g. busy server).
        # To avoid the errors, the code will try to call the api multiple times.
        # If it hit the limit in loop_limit, the code will raise an error.            
        elif is_gemini(model_name):  # google palm
            import google.generativeai as genai
            
            # store your palm key in google_api_key.txt
            google_key_path = Path("../google_api_key.txt")
            if not google_key_path.exists():
                raise FileNotFoundError(f"google_api_key.txt is not found in {google_key_path}. Please create the file and write your Google API key in the file. You can create your API at https://ai.google.dev/")
            
            google_key = easy_io.read_lines_from_txt_file(google_key_path)[0]
            genai.configure(api_key=google_key)
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
            
            loop_limit = 10
            for loop_count in range(loop_limit):
                try:
                    if is_gemini(model_name) == "palm":
                        response = genai.generate_text(**dict(parameters, prompt=prompt)).result
                    else:
                        model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
                        full_response = model.generate_content(prompt)
                        response = full_response.text
                except Exception as e:
                    if "response.prompt_feedback" in str(e):
                        print(full_response.prompt_feedback)

                    loop_process(str(e), model_name, prompt, loop_count, loop_limit, sleep_time=10)
                    continue
                
                if response is None:
                    response = ""
                
                break
            
            # palm is limited to 30 requests per minute
            if sleep_time < 0:
                time.sleep(3)
            else:
                time.sleep(sleep_time)
        elif is_claude(model_name):
            import anthropic
            
            # api key
            api_path = Path("../anthropic_api_key.txt")
            if not api_path.exists():
                raise FileNotFoundError(f"anthropic_api_key.txt is not found in {api_path}. Please create the file and write your anthropic key in the file.")
            anthropic_key = easy_io.read_lines_from_txt_file(api_path)[0]
            
            # call api
            loop_limit = 10
            for loop_count in range(loop_limit):
                try:
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    response = client.messages.create(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        **parameters,
                    ).content[0].text
                except Exception as e:
                    loop_process(str(e), model_name, prompt, loop_count, loop_limit, sleep_time=5)
                    continue
                break
                
        elif "command" in model_name:  # cohere models
            import cohere
            
            # store your cohere key in cohere_key.txt
            cohere_key_path = Path("../cohere_key.txt")
            if not cohere_key_path.exists():
                raise FileNotFoundError(f"cohere_key.txt is not found in {cohere_key_path}. Please create the file and write your cohere key in the file.")
            
            with open(cohere_key_path, "r") as f:
                cohere_key = f.read().strip()
            co = cohere.Client(cohere_key)
            
            loop_limit = 10
            for loop_count in range(loop_limit):
                try:
                    response = co.generate(**dict(parameters, prompt=prompt)).generations[0].text
                except Exception as e:
                    loop_process(str(e), model_name, prompt, loop_count, loop_limit, sleep_time=60)
                    continue
                break

            # trial key is limited to 5 calls/min
            if sleep_time < 0:
                time.sleep(12)
            else:
                time.sleep(sleep_time)
        else:
            raise ValueError(f"model_name={model_name} is not implemented")

        # cache new output
        if len(cached_output) == 0:
            dump_output_to_cache(output_dict={"prompt": prompt, "response": response}, parameters=parameters, prompt=prompt)

    return {"prompt": prompt, "response": response}
