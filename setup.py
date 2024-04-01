from setuptools import find_packages, setup

# Package metadata
name = "suql"
version = "1.0.0a"
description = "Structured and Unstructured Query Language (SUQL) Python API"
author = "Shicheng Liu"
author_email = "shicheng@cs.stanford.edu"

# Specify the packages to include. You can use `find_packages` to automatically discover them.
packages = find_packages(where="src")

# Define your dependencies
install_requires = []

# Additional package information
classifiers = [
    "Programming Language :: Python :: 3.8.17",
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