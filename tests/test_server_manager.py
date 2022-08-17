#   Copyright 2020 The GFL Authors. All Rights Reserved.
#   #
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   #
#       http://www.apache.org/licenses/LICENSE-2.0
#   #
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import unittest
import uuid

import zcommons as zc

from gfl.core.db import init_sqlite
from gfl.core.fs import FS
from gfl.core.node import GflNode
from gfl.data import *
from gfl.runtime.config import GflConfig
from gfl.runtime.manager.server_manager import ServerManager
from gfl.utils import ZipUtils

import tests.fl_job as fl_job


class ServerManagerTest(unittest.TestCase):

    def setUp(self) -> None:
        home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server-manager-home")
        fs = FS(home)
        node = GflNode.global_instance()
        config = GflConfig()
        # init
        fs.init(overwrite=True)
        node.save(fs.path.key_file())
        config.save(fs.path.config_file())
        init_sqlite(fs.path.sqlite_file())

        self.node = node
        self.server_manager = ServerManager(fs, node, config)

    def test_push_job(self):
        job_data = JobData(
            JobMeta(
                id=str(uuid.uuid4()),
                owner=self.node.address,
                create_time=zc.time.time_ms(),
                content=""
            ),
            JobConfig(
                trainer="gfl.abc.FLTrainer",
                aggregator="gfl.abc.FLAggregator"
            ),
            TrainConfig(
                model="Net",
                optimizer="SGD",
                criterion="CrossEntropyLoss"
            ),
            AggregateConfig(
                global_epoch=50
            )
        )
        package_data = ZipUtils.compress_package(fl_job)
        self.server_manager.push_job(job_data, package_data)
