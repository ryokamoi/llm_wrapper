def is_openai_model(model_name: str) -> bool:    
    if "gpt-" in model_name:
        return True
    
    if "-davinci-" in model_name:
        return True
    
    openai_model_legacy_names = ["text-curie-001", "text-babbage-001", "text-ada-001", "davinci", "curie", "babbage", "ada"]
    if model_name in openai_model_legacy_names:
        return True
    
    return False


def is_llama_model(model_name: str) -> bool:
    if "meta-llama/Llama-2-" in model_name or "lmsys/vicuna-" in model_name or "mistralai/Mixtral-" in model_name:
        return True
    
    return False
