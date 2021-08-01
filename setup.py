import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="gfl",
    version="0.2.0",
    author="malanore",
    author_email="malanore.z@gmail.com",
    description="A Galaxy Federated Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GalaxyLearning/GFL",
    project_urls={
        "Bug Tracker": "https://github.com/GalaxyLearning/GFL/issues"
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={"": ["resources/*"]},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.4",
    install_requires=[
        "web3",
        "PyYAML",
        "Flask",
        "matplotlib",
        "requests",
        "requests_toolbelt",
        "daemoniker==0.2.3",
        "ipfshttpclient",
        "numpy",
        "networkx~=2.5.1"
    ],
    extras_requires={
        "pytorch": [
            "torch>=1.4.0",
            "torchvision>=0.5.0"
        ]
    }
)
