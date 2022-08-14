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
from .. import data_pb2
from gfl.data import DatasetStatus, MIN_HOT_CNT


logger = logging.getLogger("gfl.db")


class DB(object):

    """

    """

    def __init__(self, sqlite_path):
        super(DB, self).__init__()
        self.__sqlite_path = sqlite_path
        self.__engine = create_engine(f"sqlite:///{self.__sqlite_path}")
        self.__db_session = sessionmaker(bind=self.__engine)

    def add_node(self, address, pub_key):
        """

        :param address:
        :param pub_key:
        :return:
        """
        with self.__db_session() as s:
            if s.query(NodeTable).filter_by(address=address).first() is None:
                s.add(NodeTable(address=address, pub_key=pub_key))
                s.commit()

    def get_pub_key(self, address):
        """

        :param address:
        :return:
        """
        with self.__db_session() as s:
            node = s.query(NodeTable).filter_by(address=address).one_or_none()
            return node.pub_key if node is not None else None

    def add_job(self, job: data_pb2.JobMeta):
        """

        :param job:
        :return:
        """
        with self.__db_session() as s:
            if s.query(JobTable).filter_by(job_id=job.job_id).first() is None:
                s.add(JobTable(job_id=job.job_id, owner=job.owner, create_time=job.create_time, status=job.status))
                s.add(JobTraceTable(job_id=job.job_id))
                for dataset_id in job.dataset_ids:
                    s.add(DatasetTraceTable(dataset_id=dataset_id, job_id=job.job_id))
                s.commit()

    def update_job(self, job_id, status):
        """

        :param job_id:
        :param status:
        :return:
        """
        def update_fn(job):
            job.status = status

        return 1 == self._update_one(JobTable, update_fn, job_id=job_id)

    def get_job(self, job_id):
        """

        :param job_id:
        :return:
        """
        with self.__db_session() as s:
            job = s.query(JobTable).filter_by(job_id=job_id).one_or_none()
            if job is None:
                return None
            dataset_traces = s.query(DatasetTraceTable).filter_by(job_id=job_id).all()
            dataset_ids = [dt.dataset_id for dt in dataset_traces]
            return data_pb2.JobMeta(job_id=job.job_id, owner=job.owner, create_time=job.create_time, status=job.status,
                                    dataset_ids=dataset_ids)

    def add_dataset(self, dataset: data_pb2.DatasetMeta):
        """

        :param dataset:
        :return:
        """
        with self.__db_session() as s:
            if s.query(DatasetTable).filter_by(dataset_id=dataset.dataset_id).first() is None:
                s.add(DatasetTable(dataset_id=dataset.dataset_id,
                                   owner=dataset.owner,
                                   create_time=dataset.create_time,
                                   type=dataset.type,
                                   size=dataset.size))
                s.commit()

    def update_dataset(self, dataset_id, *, inc_request_cnt=None, inc_used_cnt=None):
        """

        :param dataset_id:
        :param inc_request_cnt:
        :param inc_used_cnt:
        :return:
        """
        def update_fn(dataset):
            if inc_request_cnt is not None:
                dataset.request_cnt = dataset.request_cnt + inc_request_cnt
            if inc_used_cnt is not None:
                dataset.used_cnt = dataset.used_cnt + inc_used_cnt
                if dataset.used_cnt >= MIN_HOT_CNT:
                    dataset.status = DatasetStatus.HOT.value

        return 1 == self._update_one(DatasetTable, update_fn, dataset_id=dataset_id)

    def get_dataset(self, dataset_id):
        """

        :param dataset_id:
        :return:
        """
        with self.__db_session() as s:
            dataset = s.query(DatasetTable).filter_by(dataset_id=dataset_id).one_or_none()
            if dataset is None:
                return None
            return data_pb2.DatasetMeta(dataset_id=dataset.dataset_id,
                                        owner=dataset.owner,
                                        create_time=dataset.create_time,
                                        status=dataset.status,
                                        type=dataset.type,
                                        size=dataset.size,
                                        request_cnt=dataset.request_cnt,
                                        used_cnt=dataset.used_cnt)

    def update_dataset_trace(self, dataset_id, job_id, *, confirmed=None, score=None, keys=None):
        """

        :param dataset_id:
        :param job_id:
        :param confirmed:
        :param score:
        :param keys:
        :return:
        """
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
                         inc_aggregate_waiting_time=None,
                         inc_train_running_time=None,
                         inc_train_waiting_time=None,
                         inc_comm_time=None):
        """

        :param job_id:
        :param begin_timepoint:
        :param end_timepoint:
        :param inc_ready_time:
        :param inc_aggregate_running_time:
        :param inc_aggregate_waiting_time:
        :param inc_train_running_time:
        :param inc_train_waiting_time:
        :param inc_comm_time:
        :return:
        """
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

    def get_dataset_trace(self, dataset_id=None, job_id=None):
        """

        :param dataset_id:
        :param job_id:
        :return:
        """
        if dataset_id is None and job_id is None:
            return []
        with self.__db_session() as s:
            if dataset_id is not None and job_id is not None:
                traces = s.query(DatasetTraceTable).filter_by(dataset_id=dataset_id, job_id=job_id).all()
            elif dataset_id is None:
                traces = s.query(DatasetTraceTable).filter_by(job_id=job_id)
            else:
                traces = s.query(DatasetTraceTable).filter_by(dataset_id=dataset_id)
            ret = []
            for t in traces:
                ret.append(data_pb2.DatasetTrace(dataset_id=t.dataset_id,
                                                 job_id=t.job_id,
                                                 confirmed=t.confirmed,
                                                 score=t.score))
            return ret

    def get_job_trace(self, job_id) -> data_pb2.JobTrace:
        """

        :param job_id:
        :return:
        """
        with self.__db_session() as s:
            job_trace: JobTraceTable = s.query(JobTraceTable).filter_by(job_id=job_id).one_or_none()
            if job_trace is None:
                return None
            return data_pb2.JobTrace(job_id=job_trace.job_id,
                                     begin_timepoint=job_trace.begin_timepoint,
                                     end_timepoint=job_trace.end_timepoint,
                                     ready_time=job_trace.ready_time,
                                     aggregate_running_time=job_trace.aggregate_running_time,
                                     aggregate_waiting_time=job_trace.aggregate_waiting_time,
                                     train_running_time=job_trace.train_running_time,
                                     train_waiting_time=job_trace.train_waiting_time,
                                     comm_time=job_trace.comm_time,
                                     used_time=job_trace.used_time)

    def add_params(self, params: data_pb2.ModelParams):
        """

        :param params:
        :return:
        """
        with self.__db_session() as s:
            if s.query(ParamsTable).filter_by(job_id=params.job_id,
                                              node_address=params.node_address,
                                              dataset_id=params.dataset_id,
                                              step=params.step,
                                              is_aggregate=params.is_aggregate).first() is None:
                s.add(ParamsTable(job_id=params.job_id,
                                  node_address=params.node_address,
                                  dataset_id=params.dataset_id if params.dataset_id is not None else "",
                                  step=params.step,
                                  path=params.path,
                                  loss=params.loss,
                                  metric_name=params.metric_name,
                                  metric_value=params.metric_value,
                                  score=params.score,
                                  is_aggregate=params.is_aggregate))
                s.commit()

    def update_params(self, job_id, node_address, dataset_id, step, is_aggregate, *,
                      path=None,
                      loss=None,
                      metric_name=None,
                      metric_value=None,
                      score=None,
                      keys=None):
        """

        :param job_id:
        :param node_address:
        :param dataset_id:
        :param step:
        :param is_aggregate:
        :param path:
        :param loss:
        :param metric_name:
        :param metric_value:
        :param score:
        :param keys:
        :return:
        """
        if dataset_id is None:
            dataset_id = ""
        if keys is None:
            keys = []

        def update_fn(params):
            if path is not None or "path" in keys:
                params.path = path
            if loss is not None or "loss" in keys:
                params.loss = loss
            if metric_name is not None or "metric_name" in keys:
                params.metric_name = metric_name
            if metric_value is not None or "metric_value" in keys:
                params.metric_value = metric_value
            if score is not None or "score" in keys:
                params.score = score

        return 1 == self._update_one(ParamsTable, update_fn,
                                     job_id=job_id, node_address=node_address, dataset_id=dataset_id, step=step,
                                     is_aggregate=is_aggregate)

    def get_params(self, job_id, node_address, dataset_id, step, is_aggregate):
        """

        :param job_id:
        :param node_address:
        :param dataset_id:
        :param step:
        :param is_aggregate:
        :return:
        """
        if dataset_id is None:
            dataset_id = ""
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
                obj = s.query(entity).filter_by(**filter_dict).with_for_update().one_or_none()
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
                objs = s.query(entity).filter_by(**filter_dict).with_for_update().all()
                for obj in objs:
                    fn(obj)
                    ret += 1
                s.commit()
                return ret
        except Exception as e:
            filter_str = "&".join([f"{k}={v}" for k, v in filter_dict.items()])
            logging.error(f"An error occured when compute and update {entity.__tablename__}, filter={filter_str}")
            return 0
