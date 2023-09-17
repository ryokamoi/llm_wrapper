from setuptools import setup

setup(
    name="llm_wrapper",
    version="0.0.1",
    description=".",
    author="Ryo Kamoi",
    author_email="ryokamoi.jp@gmail.com",
    url="https://github.com/ryokamoi/llm_wrapper",
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "openai_api_wrapper @ git+ssh://git@github.com/ryokamoi/openai_api_wrapper", "easy_io @ git+ssh://git@github.com/ryokamoi/easy_io",
        "google-generativeai", "cohere"
    ]
)
