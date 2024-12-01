from setuptools import setup, find_namespace_packages

setup(
    name="medical-qa",
    version="0.1",
    packages=find_namespace_packages(include=['src*']),
    install_requires=[
        "fastapi>=0.68.0",
        "pydantic>=1.8.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "numpy>=1.21.0",
        "voyageai>=0.1.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.62.0",
        "langchain-experimental>=0.0.10",
        "langchain-community>=0.0.10",
        "uvicorn>=0.15.0",
    ],
)
