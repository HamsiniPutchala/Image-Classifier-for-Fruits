"""Setup configuration for Fruit Image Classifier project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fruit-image-classifier",
    version="1.0.0",
    author="Hamsini Putchala",
    author_email="your.email@example.com",
    description="A production-ready CNN-based fruit image classification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HamsiniPutchala/Image-Classifier-for-Fruits",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "tensorflow>=2.15.0",
        "keras>=2.15.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0.76",
        "streamlit>=1.28.1",
        "Pillow>=10.0.0",
        "python-dotenv>=1.0.0",
    ],
)