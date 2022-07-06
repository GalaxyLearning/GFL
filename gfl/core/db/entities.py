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
    "NodeTable",
    "JobTable",
    "JobTraceTable",
    "DatasetTable",
    "DatasetTraceTable",
    "ParamsTable",
    "init_sqlite"
]

from sqlalchemy import Column, String, Integer, BigInteger, Float, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base

from ..constants import JobStatus, DatasetStatus


Base = declarative_base()


class NodeTable(Base):

    __tablename__ = "node"

    address = Column("address", String(42), primary_key=True, nullable=False)
    pub_key = Column("pub_key", String(130), nullable=False)

    def __init__(self, address, pub_key):
        super(NodeTable, self).__init__()
        self.address = address
        self.pub_key = pub_key


class JobTable(Base):

    __tablename__ = "job"

    job_id = Column("job_id", String(38), primary_key=True, nullable=False)
    owner = Column("owner", String(42), nullable=False)
    create_time = Column("create_time", BigInteger, nullable=False)
    status = Column("status", Integer, nullable=False, default=JobStatus.NEW.value)

    def __init__(self, job_id, owner, create_time, status=JobStatus.NEW.value):
        super(JobTable, self).__init__()
        self.job_id = job_id
        self.owner = owner
        self.create_time = create_time
        self.status = status


class JobTraceTable(Base):

    __tablename__ = "job_trace"

    job_id = Column("job_id", String(38), primary_key=True, nullable=False)
    begin_timepoint = Column("begin_timepoint", BigInteger, nullable=False, default=0)
    end_timepoint = Column("end_timepoint", BigInteger, nullable=False, default=0)
    ready_time = Column("ready_time", Integer, nullable=False, default=0)
    aggregate_running_time = Column("aggregate_running_time", Integer, nullable=False, default=0)
    aggregate_waiting_time = Column("aggregate_waiting_time", Integer, nullable=False, default=0)
    train_running_time = Column("train_running_time", Integer, nullable=False, default=0)
    train_waiting_time = Column("train_waiting_time", Integer, nullable=False, default=0)
    comm_time = Column("comm_time", Integer, nullable=False, default=0)
    used_time = Column("used_time", Integer, nullable=False, default=0)

    def __init__(self, job_id,
                 begin_timepoint=0,
                 end_timepoint=0,
                 ready_time=0,
                 aggregate_running_time=0,
                 aggregate_waiting_time=0,
                 train_running_time=0,
                 train_waiting_time=0,
                 comm_time=0,
                 used_time=0):
        super(JobTraceTable, self).__init__()
        self.job_id = job_id
        self.begin_timepoint = begin_timepoint
        self.end_timepoint = end_timepoint
        self.ready_time = ready_time
        self.aggregate_running_time = aggregate_running_time
        self.aggregate_waiting_time = aggregate_waiting_time
        self.train_running_time = train_running_time
        self.train_waiting_time = train_waiting_time
        self.comm_time = comm_time
        self.used_time = used_time


class DatasetTable(Base):

    __tablename__ = "dataset"

    dataset_id = Column("dataset_id", String(38), primary_key=True, nullable=False)
    owner = Column("owner", String(42), nullable=False)
    create_time = Column("create_time", BigInteger, nullable=False)
    type = Column("type", Integer, nullable=False)
    size = Column("size", BigInteger, nullable=False)
    status = Column("status", Integer, nullable=False, default=DatasetStatus.NEW.value)
    request_cnt = Column("request_cnt", Integer, nullable=False, default=0)
    used_cnt = Column("used_cnt", Integer, nullable=False, default=0)

    def __init__(self, dataset_id, owner, create_time, type, size,
                 status=DatasetStatus.NEW.value, request_cnt=0, used_cnt=0):
        super(DatasetTable, self).__init__()
        self.dataset_id = dataset_id
        self.owner = owner
        self.create_time = create_time
        self.type = type
        self.size = size
        self.status = status
        self.request_cnt = request_cnt
        self.used_cnt = used_cnt


class DatasetTraceTable(Base):

    __tablename__ = "dataset_trace"

    id = Column("id", Integer, primary_key=True, nullable=False, auto_increment="auto")
    dataset_id = Column("dataset_id", String(38), nullable=False)
    job_id = Column("job_id", String(38), nullable=False)
    confirmed = Column("confirmed", Boolean, nullable=False, default=False)
    score = Column("score", Float, nullable=False, default=0)

    def __init__(self, dataset_id, job_id, confirmed=False, score=0):
        super(DatasetTraceTable, self).__init__()
        self.dataset_id = dataset_id
        self.job_id = job_id
        self.confirmed = confirmed
        self.score = score


class ParamsTable(Base):

    __tablename__ = "params"

    id = Column("id", Integer, primary_key=True, nullable=False, auto_increment="auto")
    job_id = Column("job_id", String(38), nullable=False)
    node_address = Column("node_address", String(42), nullable=False)
    dataset_id = Column("dataset_id", String(38), nullable=True)
    step = Column("step", Integer, nullable=False)
    path = Column("path", String(1024), nullable=False)
    loss = Column("loss", Float, nullable=False)
    metric_name = Column("metric_name", String(64), nullable=False)
    metric_value = Column("metric_value", Float, nullable=False)
    score = Column("score", Float, nullable=False)
    is_aggregate = Column("is_aggregate", Boolean, nullable=False, default=False)

    def __init__(self,
                 job_id,
                 node_address,
                 dataset_id,
                 step,
                 path,
                 loss,
                 metric_name,
                 metric_value,
                 score,
                 is_aggregate=False):
        super(ParamsTable, self).__init__()
        self.job_id = job_id
        self.node_address = node_address
        self.dataset_id = dataset_id
        self.step = step
        self.path = path
        self.loss = loss
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.score = score
        self.is_aggregate = is_aggregate


def init_sqlite(path):
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
