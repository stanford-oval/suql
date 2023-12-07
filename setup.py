from setuptools import find_packages, setup

# Package metadata
name = "suql"
version = "0.1.0"
description = "Structured and Unstructured Query Language (SUQL): formal executable representation that naturally covers compositions of structured and unstructured data queries"
author = "Shicheng Liu, Jialiang Xu, Wesley Tjangnaka, Sina J. Semnani, Chen Jie Yu, Gui Dávid, Monica S. Lam"
author_email = "sliu22@stanford.edu"

# Specify the packages to include. You can use `find_packages` to automatically discover them.
packages = find_packages(where="src")

# Define your dependencies
install_requires = []

# Additional package information
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
]

# Call setup() with package information
setup(
    name=name,
    version=version,
    description=description,
    author=author,
    author_email=author_email,
    packages=packages,
    package_dir={"": "src"},
    install_requires=install_requires,
    classifiers=classifiers,
)
