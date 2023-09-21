from setuptools import setup

setup(
    name="llm_wrapper",
    version="0.1.1",
    description="This project includes an unofficial wrapper for LLM APIs.",
    author="Ryo Kamoi",
    author_email="ryokamoi.jp@gmail.com",
    url="https://github.com/ryokamoi/llm_wrapper",
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "openai_api_wrapper @ git+ssh://git@github.com/ryokamoi/openai_api_wrapper@0.4.0", "easy_io @ git+ssh://git@github.com/ryokamoi/easy_io@0.1.1",
        "google-generativeai", "cohere"
    ]
)
