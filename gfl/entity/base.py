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


from gfl.utils.json_utils import JsonUtil


class BaseEntity(object):

    def __init__(self, **kwargs):
        super(BaseEntity, self).__init__()

        cls = type(self)

        for k in cls.__dict__.keys():
            setattr(self, k, None)

        for k, v in kwargs.items():
            if k in cls.__dict__:
                setattr(self, k, v)

    def __repr__(self):
        return JsonUtil.to_json(self)

    def __str__(self):
        return self.__repr__()