from setuptools import setup, find_packages

setup(
    name="medrgb",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers==4.42.3",
        "sentence-transformers==2.2.2",
        "faiss-cpu==1.7.4",
        "python-liquid==1.10.2",
        "openai>=1.0.0",
        "tiktoken==0.6.0",
        "tqdm==4.66.1",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
