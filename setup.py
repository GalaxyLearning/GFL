#  Copyright 2020 The GFL Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

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
    install_requires=[

    ],
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
