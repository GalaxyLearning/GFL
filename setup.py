import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = [line.strip() for line in f.readlines()]
    requires = [r for r in requires if r != ""]

setuptools.setup(
    name="gfl_p",
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
    python_requires=">=3.6",
    install_requires=requires,
    extras_requires={
        "pytorch": [
            "torch>=1.4.0",
            "torchvision>=0.5.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "gfl_p=gfl_p.shell.ipython:startup"
        ]
    }
)
