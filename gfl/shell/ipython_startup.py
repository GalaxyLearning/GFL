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

__all__ = [
    "gfl",
    "net",
    "node",
    "version"
]

import os
import time
import warnings

import gfl
from gfl.api.net import Net
from gfl.api.node import Node
from gfl.core.fs.path import Path


_path = Path(os.environ["__GFL_HOME__"])

try:
    net = Net(_path.home())
except:
    warnings.warn(f"Server node has not started, 'net' object is unavailable")
    net = None

node = Node(_path.home())

version = "0.2.0"


def welcome():
    global version
    print(f"GFL {version} ({time.asctime()})")


welcome()
