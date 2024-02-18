from setuptools import setup

setup(
    name="llm_wrapper",
    version="0.2.3",
    description="This project includes an unofficial wrapper for LLM APIs.",
    author="Ryo Kamoi",
    author_email="ryokamoi.jp@gmail.com",
    url="https://github.com/ryokamoi/llm_wrapper",
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "openai==0.28",  # openai models
        "transformers==4.34.1", "easyllm==0.6.2", "langchain==0.0.350", "accelerate==0.25.0", "sentencepiece==0.1.99",  # llama2
        "easy_io @ git+ssh://git@github.com/ryokamoi/easy_io@0.2.1",
        "google-generativeai==0.3.2", "cohere==4.45"
    ]
)
