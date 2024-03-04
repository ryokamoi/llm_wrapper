from typing import Union
from pathlib import Path
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenModel():
    def __init__(self, model_name_or_path: str, cache_dir: Union[str, Path, None] = None):
        self.model_name_or_path = model_name_or_path
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @staticmethod
    def convert_parameters(parameters: dict) -> dict:
        parameters = copy.deepcopy(parameters)
        
        if "model" in parameters.keys():
            del parameters["model"]
        
        if "max_tokens" in parameters.keys():
            parameters["max_new_tokens"] = parameters["max_tokens"]
            del parameters["max_tokens"]

        return parameters
    
    def __call__(self, prompt: str, stop: list[str] = None, parameters: dict = {"temperature": 0.5, "max_new_tokens": 4096}):
        parameters = self.convert_parameters(parameters)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pad_token_id=self.tokenizer.eos_token_id,
            **parameters
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response


def load_open_model(model_name: dict):
    with open("../huggingface_model_path.txt", "r") as f:
        huggingface_path = f.read()
    
    return OpenModel(model_name_or_path=model_name, cache_dir=huggingface_path)
