from setuptools import setup, find_packages


setup(
    name="cloth",
    version="0.1.0",
    description="A better solution to RAG",
    author="Soami Kapadia",
    author_email="kapadiasoami@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/skap3214/cloth",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license="MIT",
    python_requires='>=3.10',
    install_requires=[
        "langchain-openai>=0.1.3",
        "langchain-community>=0.0.38",
        "langchain-chroma>=0.1.1",
        "langchain-groq>=0.1.3",
        "langchain>=0.1.1.6",
        "langchain-pinecone>=0.1.1",
    ]
)