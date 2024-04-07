from typing import Optional, Literal, TypedDict
import time
from pathlib import Path

import openai


# ParamsType = dict[str, Union[str, float]]
class ParamsType(TypedDict):
    prompt: str
    temperature: float
    messages: list[dict[str, str]]


def get_chat_parameters(prompt: str, parameters: ParamsType) -> ParamsType:
    return dict(parameters, messages=[{"role": "user", "content": prompt}])


def get_edit_parameters(input_sentence: str, instruction: str, parameters: ParamsType):
    return dict(input=input_sentence, instruction=instruction, **parameters)


def openai_text_api(mode: Literal["complete", "chat", "edit"], parameters: ParamsType,
                    openai_api_key_path: Optional[Path]=Path("../openai_api_key.txt"),
                    organization: Optional[str] = None):
    """OpenAI API wrapper for text completion, chat, and edit."""
    
    if organization is not None:
        openai.organization = organization
    
    if openai_api_key_path is None or not openai_api_key_path.exists():
        raise ValueError(f"{openai_api_key_path} does not exist.")
    
    with open(openai_api_key_path, "r") as f:
        openai.api_key = f.read().strip()
    
    # prompt
    if mode == "chat":
        prompt = parameters["messages"][-1]["content"]
    elif mode == "complete":
        prompt = parameters["prompt"]
    else:
        raise NotImplementedError()
    
    # openai api
    response = None
    for try_count in range(10):
        try:
            if mode == "chat":
                response = openai.ChatCompletion.create(**parameters)
            elif mode == "complete":
                response = openai.Completion.create(**parameters)
            elif mode == "edit":
                raise NotImplementedError()
            else:
                raise ValueError(f"{mode} is not a valid value for the mode parameter. Please choose from 'chat', 'complete', or 'edit'.")
        except Exception as e:
            print("Exception occurred in OpenAI API:\n")
            print(str(e))
            if "This model's maximum context length is" in str(e):
                response = None
                break
            
            if try_count == 9:
                raise Exception("OpenAI API failed for 10 times for this example. Please try again later.")
            
            sleep_seconds = 5
            print(f"\nRetrying after {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            
            continue
        break
    
    # output dict
    output_dict = {
        "prompt": prompt, "response": response,
    }
    
    return output_dict
