from typing import List, Optional, Any

import torch
from transformers import AutoTokenizer
from langchain.llms.base import LLM


class Llama2(LLM):
    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 2048
    do_sample: bool = True
    model_name_or_path: str = None
    tokenizer: Any
    model: Any

    def __init__(self, model_name_or_path, model_path=None, temperature=0.1, do_sample=True, max_tokens=2048, bit4=False):
        super().__init__()
        
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_tokens = max_tokens
        self.model_name_or_path = model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if bit4 == False:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=False,
                cache_dir=model_path,
            )
            self.model.eval()
        else:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=model_path,
            )
            self.model.eval()

    @property
    def _llm_type(self) -> str:
        return "Llama2"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_tokens,
            "do_sample": self.do_sample,
            "top_k": 50 if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "temperature": self.temperature if self.do_sample else None,
            "repetition_penalty": 1.2 if self.do_sample else None,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generate_ids = self.model.generate(**generate_input)
        generate_ids = [item[len(input_ids[0]) : -1] for item in generate_ids]
        result_message = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return result_message


default_dict = {
    "model_name": "meta-llama/Llama-2-70b-chat-hf",
    "temperature": 0.5,
    "max_tokens": None,
    "do_sample": True,
}


def load_llama2(llama2_parameters: dict = default_dict) -> Llama2:
    with open("../huggingface_model_path.txt", "r") as f:
        llama2_path = f.read()
    
    if "do_sample" == False:
        llama2_parameters["temperature"] = None
    
    # change key model_name to model_name_or_path
    llama2_parameters["model_name_or_path"] = llama2_parameters["model_name"]
    del llama2_parameters["model_name"]
    
    return Llama2(model_path=llama2_path, **llama2_parameters)
