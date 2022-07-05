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
    "DB"
]

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .entities import *
from .. import data_pb2, constants

logger = logging.getLogger("gfl.db")


class DB(object):

    def __init__(self, sqlite_path):
        super(DB, self).__init__()
        self.__sqlite_path = sqlite_path
        self.__engine = create_engine(f"sqlite:///{self.__sqlite_path}")
        self.__db_session = sessionmaker(bind=self.__engine)

    def add_node(self, address, pub_key):
        with self.__db_session() as s:
            if s.query(NodeTable).filter_by(address=address).first() is None:
                s.add(NodeTable(address=address, pub_key=pub_key))
                s.commit()

    def get_pub_key(self, address):
        with self.__db_session() as s:
            node = s.query(NodeTable).filter_by(address=address).one_or_none()
            return node.pub_key if node is not None else None

    def add_job(self, job: data_pb2.JobMeta):
        with self.__db_session() as s:
            if s.query(JobTable).filter_by(job_id=job.id).first() is None:
                s.add(JobTable(job_id=job.job_id, owner=job.owner, create_time=job.create_time, status=job.status))
                s.add(JobTraceTable(job_id=job.job_id))
                for dataset_id in job.dataset_ids:
                    s.add(DatasetTraceTable(dataset_id=dataset_id, job_id=job.job_id))
                s.commit()

    def update_job(self, job_id, status):
        def update_fn(job):
            job.status = status

        return 1 == self._update_one(JobTable, update_fn, job_id=job_id)

    def add_dataset(self, dataset: data_pb2.DatasetMeta):
        with self.__db_session() as s:
            if s.query(DatasetTable).filter_by(dataset_id=dataset.dataset_id).first() is None:
                s.add(DatasetTable(dataset_id=dataset.dataset_id,
                                   owner=dataset.owner,
                                   create_time=dataset.create_time,
                                   type=dataset.type,
                                   size=dataset.size))

    def update_dataset(self, dataset_id, *, inc_request_cnt=None, inc_used_cnt=None):
        def update_fn(dataset):
            if inc_request_cnt is not None:
                dataset.request_cnt = dataset_id.request_cnt + inc_request_cnt
            if inc_used_cnt is not None:
                dataset.used_cnt = dataset.used_cnt + inc_used_cnt
                if dataset.used_cnt >= constants.MIN_HOT_CNT:
                    dataset.status = constants.DatasetStatus.HOT.value

        return 1 == self._update_one(DatasetTable, update_fn, dataset_id=dataset_id)

    def update_dataset_trace(self, dataset_id, job_id, *, confirmed=None, score=None, keys=None):
        if keys is None:
            keys = []

        def update_fn(dataset):
            if confirmed is not None or "confirmed" in keys:
                dataset.confirmed = confirmed
            if score is not None or "score" in keys:
                dataset.score = score

        return 1 == self._update_one(DatasetTraceTable, update_fn, dataset_id=dataset_id, job_id=job_id)

    def update_job_trace(self, job_id, *,
                         begin_timepoint=None,
                         end_timepoint=None,
                         inc_ready_time=None,
                         inc_aggregate_running_time=None,
                         inc_aggregate_waiting_time,
                         inc_train_running_time,
                         inc_train_waiting_time,
                         inc_comm_time):

        def update_fn(job_trace):
            if begin_timepoint is not None:
                job_trace.begin_timepoint = begin_timepoint
            if end_timepoint is not None:
                job_trace.end_timepoint = end_timepoint
                job_trace.used_time = (job_trace.end_timepoint - job_trace.begin_timepoint) // 1000

            if inc_ready_time is not None:
                job_trace.ready_time = job_trace.ready_time + inc_ready_time
            if inc_aggregate_running_time is not None:
                job_trace.aggregate_running_time = job_trace.aggregate_running_time + inc_aggregate_running_time
            if inc_aggregate_waiting_time is not None:
                job_trace.aggregate_waiting_time = job_trace.aggregate_waiting_time + inc_aggregate_waiting_time
            if inc_train_running_time is not None:
                job_trace.train_running_time = job_trace.train_running_time + inc_train_running_time
            if inc_train_waiting_time is not None:
                job_trace.train_waiting_time = job_trace.train_waiting_time + inc_train_waiting_time
            if inc_comm_time is not None:
                job_trace.comm_time = job_trace.comm_time + inc_comm_time

        return 1 == self._update_one(JobTraceTable, update_fn, job_id=job_id)

    def add_params(self, params: data_pb2.ModelParams):
        with self.__db_session() as s:
            if s.query(ParamsTable).filter_by(job_id=params.job_id,
                                              node_address=params.node_address,
                                              dataset_id=params.dataset_id,
                                              step=params.step,
                                              is_aggregate=params.is_aggregate).first() is None:
                s.add(ParamsTable(job_id=params.job_id,
                                  node_address=params.node_address,
                                  dataset_id=params.dataset_id,
                                  step=params.step,
                                  path=params.path,
                                  loss=params.loss,
                                  metric_name=params.metric_name,
                                  metric_value=params.metric_value,
                                  score=params.score,
                                  is_aggregate=params.is_aggregate))
                s.commit()

    def get_params(self, job_id, node_address, dataset_id, step, is_aggregate):
        with self.__db_session() as s:
            params = s.query(ParamsTable).filter_by(job_id=job_id,
                                                    node_address=node_address,
                                                    dataset_id=dataset_id,
                                                    step=step,
                                                    is_aggregate=is_aggregate).one_or_none()
            if params is None:
                return None
            return data_pb2.ModelParams(job_id=params.job_id,
                                        node_address=params.node_address,
                                        dataset_id=params.dataset_id,
                                        step=params.step,
                                        path=params.path,
                                        loss=params.loss,
                                        metric_name=params.metric_name,
                                        metric_value=params.metric_value,
                                        score=params.score,
                                        is_aggregate=params.is_aggregate)

    def _update_one(self, entity, fn, **filter_dict):
        try:
            with self.__db_session() as s:
                ret = 0
                obj = s.query(entity).filter_by(**filter_dict).wait_for_update().one_or_none()
                if obj is not None:
                    fn(obj)
                    ret += 1
                s.commit()
                return ret
        except Exception as e:
            filter_str = "&".join([f"{k}={v}" for k, v in filter_dict.items()])
            logging.error(f"An error occured when compute and update {entity.__tablename__}, filter={filter_str}")
            return 0

    def _update_all(self, entity, fn, **filter_dict):
        try:
            with self.__db_session() as s:
                ret = 0
                objs = s.query(entity).filter_by(**filter_dict).wait_for_update().all()
                for obj in objs:
                    fn(obj)
                    ret += 1
                s.commit()
                return ret
        except Exception as e:
            filter_str = "&".join([f"{k}={v}" for k, v in filter_dict.items()])
            logging.error(f"An error occured when compute and update {entity.__tablename__}, filter={filter_str}")
            return 0
