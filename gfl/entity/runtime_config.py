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


WAITING_JOB_LIST = list()
PENDING_JOB_LIST = list()
EXEC_JOB_LIST = list()
CONNECTED_TRAINER_LIST = list()
WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST = list()


def get_waiting_job():
    return WAITING_JOB_LIST


def add_waiting_job(job):
    WAITING_JOB_LIST.append(job)


def remove_waiting_job(job):
    WAITING_JOB_LIST.remove(job)


def get_pending_job():
    return PENDING_JOB_LIST


def add_pending_job(job):
    PENDING_JOB_LIST.append(job)


def remove_pending_job(job):
    PENDING_JOB_LIST.remove(job)


def add_exec_job(job):
    EXEC_JOB_LIST.put(job)


def get_exec_job():
    return EXEC_JOB_LIST.get()
