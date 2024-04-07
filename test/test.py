import unittest

from llm_wrapper import llm_api
import llm_wrapper.open_models


test_prompt = "generate five random alphabets"

class TestCase(unittest.TestCase):
    def test_openai(self):
        model_name = "gpt-3.5-turbo"
        response1 = llm_api(model_name=model_name, prompt=test_prompt, overwrite_cache=True)  # dump cache
        response2 = llm_api(model_name=model_name, prompt=test_prompt)  # read cache
        assert response1 == response2
        
        response1_temp = llm_api(model_name=model_name, prompt=test_prompt, updated_parameters={"temperature": 0.5}, overwrite_cache=True)
        response2_temp = llm_api(model_name=model_name, prompt=test_prompt, updated_parameters={"temperature": 0.5})  # read cache
        assert response1_temp == response2_temp
    
    def test_llama(self):
        model_name = "meta-llama/Llama-2-13b-chat-hf"
        loaded_model = llm_wrapper.open_models.load_open_model(model_name)
        response1_llama = llm_api(model_name=model_name, prompt=test_prompt, updated_parameters={"temperature": 0.5}, overwrite_cache=True,
                                  loaded_model=loaded_model)  # dump cache
        response2_llama = llm_api(model_name=model_name, prompt=test_prompt, updated_parameters={"temperature": 0.5},
                                  loaded_model=loaded_model)  # read cache
        assert response1_llama == response2_llama


if __name__ == "__main__":
    unittest.main()
