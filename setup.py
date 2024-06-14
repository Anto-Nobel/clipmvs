from setuptools import setup, find_packages

setup(
    name="clipmvs",
    version="0.1.0",
    description="A library for processing video frames and storing CLIP embeddings in Qdrant",
    author="Vijay Anand Kandaswamy, Anto Nobel",
    url="https://github.com/Anto-Nobel/clipmvs",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "qdrant-client",
        "opencv-python",
        "Pillow",
        "matplotlib",
        "protobuf"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
