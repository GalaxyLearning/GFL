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
    "JobStatus",
    "DatasetStatus",
    "DatasetType",
    "MIN_HOT_CNT"
]

from enum import Enum


class JobStatus(Enum):

    # just created job
    NEW = 0

    # for aggregate : send all requests to train nodes
    # for train     : send confirm response and waiting others
    READY = 1

    # for aggregate : running aggregate algorithm
    # for train     : running train algorithm
    RUNNING = 2

    # for aggregate : waiting the local update params of train nodes
    # for train     : waiting aggregated params of aggregate node
    WAITING = 3

    # job finished
    FINISHED = 4

    # job not finished and terminated by user or error
    ABORTED = 5


class DatasetStatus(Enum):

    # just published dataset, not used(also means validated) by others
    NEW = 0

    # the dataset has been used by others at least once, this means it is validated by others.
    USED = 1

    # the dataset has been used by others at least ten times
    HOT = 2

    # the dataset has been destroyed by its owner
    DESTROYED = 3


class DatasetType(Enum):

    IMAGE = 0
    VIDEO = 1
    AUDIO = 2
    TEXT = 3
    STRUCTURE = 4


MIN_HOT_CNT = 10
