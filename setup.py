from setuptools import find_packages, setup

# Package metadata
name = "suql"
version = "1.0.0a3"
description = "Structured and Unstructured Query Language (SUQL) Python API"
author = "Shicheng Liu"
author_email = "shicheng@cs.stanford.edu"
url = "https://github.com/stanford-oval/suql"

# Specify the packages to include. You can use `find_packages` to automatically discover them.
packages = find_packages(where="src")

# Define your dependencies
install_requires = [
    'openai==1.3.2',
    'Jinja2==3.1.2',
    'Flask==2.3.2',
    'Flask-Cors==4.0.0',
    'Flask-RESTful==0.3.10',
    'transformers==4.38.2',
    'torch==2.0.1',
    'requests==2.31.0',
    'spacy==3.7.4',
    'tiktoken==0.4.0',
    'psycopg2-binary==2.9.7',
    'pglast==5.3',
    'FlagEmbedding==1.2.5',
]

# Additional package information
classifiers = [
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
]

# Call setup() with package information
setup(
    name=name,
    version=version,
    description=description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author=author,
    author_email=author_email,
    packages=packages,
    package_dir={"": "src"},
    install_requires=install_requires,
    url=url,
    classifiers=classifiers,
    package_data={
        "": ["*.prompt"]
    }
)