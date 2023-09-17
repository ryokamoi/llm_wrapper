# LLM Wrapper

This is an unofficial wrapper for LLM APIs.

* OpenAI models
* Google Palm
* Cohere Commands

## Setup

```sh
pip install git+https://github.com/ryokamoi/llm_wrapper
```

## Example

```python
from llm_wrapper import llm_api
print(llm_api("gpt-3.5-turbo", prompt="Translate this sentence into Japanese: I am GPT-3.5."))
# {'prompt': 'Translate this sentence into Japanese: I am GPT-3.5.', 'response': '私はGPT-3.5です。 (Watashi wa GPT-3.5 desu.)'}

# When temperature=0. (default), this wrapper will store and read cached results.
llm_api("gpt-3.5-turbo", prompt="Translate this sentence into Japanese: I am GPT-3.5.")
# read cache from llm_cache/554a571588d632d02bd8cc330fee66fea40adf7d242763b462a7691ee858afd4d3a852caaa02058db5e9810edac0fee5021754c0f4479842d374e915a7cb21c0.json
# {'prompt': 'Translate this sentence into Japanese: I am GPT-3.5.', 'response': '私はGPT-3.5です。 (Watashi wa GPT-3.5 desu.)'}
```

* You can overwrite the cache by setting `cache_overwrite=True`.
* You can overwrite the temperature by setting `updated_parameters={"temperature": 0.5}`. You can update any parameters in the same way.
