from setuptools import setup

setup(
    name="wikindex",
    version="0.1.0",
    packages=["wikindex"],
    install_requires=[
        "torch==2.8.0+cu126",
        "transformers==4.56.2",
        "pydantic==2.11.9"
    ],
)