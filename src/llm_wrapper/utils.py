def is_openai_model(model_name: str) -> bool:    
    if "gpt-" in model_name:
        return True
    
    if "-davinci-" in model_name:
        return True
    
    openai_model_legacy_names = ["text-curie-001", "text-babbage-001", "text-ada-001", "davinci", "curie", "babbage", "ada"]
    if model_name in openai_model_legacy_names:
        return True
    
    return False


def is_gemini(model_name: str) -> tuple[str, bool]:
    if model_name in ["text-bison-001"]:
        return "palm"
    elif "gemini-" in model_name:
        return "gemini"
    
    return False


def is_claude(model_name: str) -> bool:
    if "claude-" in model_name:
        return True
    
    return False


def is_open_model(model_name: str) -> bool:
    if is_llama_model(model_name):
        return True
    
    if is_qwen_model(model_name):
        return True
    
    if is_gemma(model_name):
        return True

    if "mistralai/Mixtral-" in model_name:
        return True
    
    return False


def is_llama_model(model_name: str) -> bool:
    if "meta-llama/Llama-2-" in model_name:
        return True
    
    return False


def is_qwen_model(model_name: str) -> bool:
    if "Qwen/Qwen" in model_name:
        return True
    
    return False


def is_gemma(model_name: str) -> bool:
    if "google/gemma" in model_name:
        return True

    return False
