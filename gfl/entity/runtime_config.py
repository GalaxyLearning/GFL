# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gfl.entity.base import BaseEntity

class RuntimeConfig(BaseEntity):

    def __init__(self):
        super(RuntimeConfig, self).__init__()


class RuntimeServerConfig(RuntimeConfig):

    CONNECTED_TRAINER_LIST = (list, str)
    WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST = (list, str)

    def __init__(self):
        super(RuntimeServerConfig, self).__init__()


class RuntimeClientConfig(RuntimeConfig):

    JOB_FINISHED_LIST = (list, str)

    def __init__(self):
        super(RuntimeClientConfig, self).__init__()