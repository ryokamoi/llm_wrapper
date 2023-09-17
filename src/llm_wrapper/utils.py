def is_this_openai_model(model_name: str) -> bool:    
    if "gpt-" in model_name:
        return True
    
    if "-davinci-" in model_name:
        return True
    
    openai_model_legacy_names = ["text-curie-001", "text-babbage-001", "text-ada-001", "davinci", "curie", "babbage", "ada"]
    if model_name in openai_model_legacy_names:
        return True
    
    return False


def is_this_model_for_chat(model_name: str) -> bool:
    if "gpt-" in model_name:
        return True
    
    return False
