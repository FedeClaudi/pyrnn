from setuptools import setup, find_namespace_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "rich>=9.0.0",
    "pyinspect",
    "matplotlib",
    "seaborn",
    "torch",
    "numpy",
    "networkx",
    "myterial",
]

setup(
    name="pyrnn",
    version="0.0.2",
    description="Pytorch implementation of vanilla RNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={"dev": ["coverage-badge"]},
    python_requires=">=3.6,",
    packages=find_namespace_packages(),
    include_package_data=True,
    url="https://github.com/FedeClaudi/pyrnn",
    author="Federico Claudi",
    zip_safe=False,
    entry_points={},
)
