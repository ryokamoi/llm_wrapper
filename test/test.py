import unittest

from llm_wrapper import llm_api


test_prompt = "Hello, I am a human."

class TestCase(unittest.TestCase):
    def test_openai(self):
        _ = llm_api(model_name="gpt-3.5-turbo", prompt=test_prompt, overwrite_cache=True)  # dump cache
        _ = llm_api(model_name="gpt-3.5-turbo", prompt=test_prompt)  # read cache
        
        _ = llm_api(model_name="gpt-3.5-turbo", prompt=test_prompt, updated_parameters={"temperature": 0.5}, overwrite_cache=True)
        _ = llm_api(model_name="gpt-3.5-turbo", prompt=test_prompt, updated_parameters={"temperature": 0.5})  # read cache
    
    def test_cohere(self):
        _ = llm_api(model_name="command", prompt=test_prompt, overwrite_cache=True)  # dump cache
        _ = llm_api(model_name="command", prompt=test_prompt)  # read cache

        _ = llm_api(model_name="command", prompt=test_prompt, updated_parameters={"temperature": 0.5}, overwrite_cache=True)
    
    # def test_palm(self):
    #     _ = llm_api(model_name="text-bison-001", prompt=test_prompt)  # dump cache
    #     _ = llm_api(model_name="text-bison-001", prompt=test_prompt)  # read cache

    #     _ = llm_api(model_name="text-bison-001", prompt=test_prompt, updated_parameters={"temperature": 0.5})


if __name__ == "__main__":
    unittest.main()
