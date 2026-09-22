from setuptools import setup, find_packages

setup(
    name="navigators-idr",
    version="1.0.0",
    description="AI/ML-powered Intelligent Dead Reckoning for GNSS-denied navigation",
    author="Navigators",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "matplotlib>=3.7.0",
        "filterpy>=1.4.5",
        "geopy>=2.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "tensorboard>=2.13.0"],
        "edge": ["onnx>=1.14.0", "onnxruntime>=1.15.0"],
    },
)
