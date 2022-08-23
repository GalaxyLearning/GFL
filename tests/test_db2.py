class DBTest(unittest.TestCase):

    def setUp(self) -> None:
        sqlite_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "test.sqlite")
        if os.path.exists(sqlite_filepath):
            os.remove(sqlite_filepath)
        init_sqlite(sqlite_filepath)
        self.db = DB(sqlite_filepath)

    def test_node(self):
        node = GflNode.new_node()
        self.db.add_node(node.address, node.pub_key)
        self.assertEqual(self.db.get_pub_key(node.address), node.pub_key)

    def test_job(self):
        job_id, owner, create_time, dataset_ids = self._add_job()
        self.db.update_job(job_id, constants.JobStatus.RUNNING.value)
        job = self.db.get_job(job_id)
        self.assertEqual(job.job_id, job_id)
        self.assertEqual(job.owner, owner)
        self.assertEqual(job.create_time, create_time)
        self.assertListEqual(list(job.dataset_ids), dataset_ids)

    def test_dataset(self):
        node = GflNode.new_node()
        dataset_id = str(uuid.uuid4())
        owner = node.address
        create_time = zc.time.time_ms()
        status = constants.DatasetStatus.NEW.value
        type = constants.DatasetType.IMAGE.value
        size = zc.units.BinaryUnits.B.convert_from(10, zc.units.BinaryUnits.MiB)
        request_cnt = 0
        used_cnt = 0
        self.db.add_dataset(data_pb2.DatasetMeta(dataset_id=dataset_id,
                                                 owner=owner,
                                                 create_time=create_time,
                                                 status=status,
                                                 type=type,
                                                 size=size,
                                                 request_cnt=request_cnt,
                                                 used_cnt=used_cnt))
        self.assertTrue(self.db.update_dataset(dataset_id, inc_request_cnt=32, inc_used_cnt=12))
        dataset = self.db.get_dataset(dataset_id)
        self.assertEqual(dataset.owner, owner)
        self.assertEqual(dataset.status, constants.DatasetStatus.HOT.value)
        self.assertEqual(dataset.request_cnt, 32)
        self.assertEqual(dataset.used_cnt, 12)

    def test_job_trace(self):
        job_id, _, _, _ = self._add_job()
        job_trace = self.db.get_job_trace(job_id)
        self.assertEqual(job_trace.job_id, job_id)
        self.assertEqual(job_trace.ready_time, 0)

        timepoint = zc.time.time_ms()
        self.assertTrue(self.db.update_job_trace(job_id, end_timepoint=timepoint, inc_ready_time=10, inc_comm_time=20))
        job_trace = self.db.get_job_trace(job_id)
        self.assertEqual(job_trace.end_timepoint, timepoint)
        self.assertEqual(job_trace.comm_time, 20)
        self.assertEqual(job_trace.ready_time, 10)

    def test_dataset_trace(self):
        job_id, _, _, dataset_ids = self._add_job()
        self.db.update_dataset_trace(dataset_ids[0], job_id, confirmed=True, score=78)
        d_t_1 = self.db.get_dataset_trace(dataset_ids[0], job_id)[0]
        self.assertTrue(d_t_1.confirmed)
        self.assertEqual(round(1000000 * d_t_1.score), 78000000)

    def test_params(self):
        node = GflNode.new_node()
        job_id = str(uuid.uuid4())
        params = data_pb2.ModelParams(job_id=job_id,
                                      node_address=node.address,
                                      step=3,
                                      path="ipfs://<hash-address>",
                                      loss=1.09,
                                      metric_name="acc",
                                      metric_value=0.89,
                                      score=34.5,
                                      is_aggregate=True)
        self.db.add_params(params)
        self.db.update_params(job_id=job_id, node_address=node.address, dataset_id=None, step=3, is_aggregate=True,
                              loss=0.98, score=54)
        p = self.db.get_params(job_id=job_id, node_address=node.address, dataset_id=None, step=3, is_aggregate=True)
        self.assertEqual(round(1000000 * p.loss), 980000)
        self.assertEqual(round(1000000 * p.score), 54000000)

    def _add_job(self):
        node = GflNode.new_node()
        job1_id = str(uuid.uuid4())
        job2_id = str(uuid.uuid4())
        owner = node.address
        create_time = zc.time.time_ms()
        dataset_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        self.db.add_job(data_pb2.JobMeta(job_id=job_id, owner=owner, create_time=create_time,
                                         status=constants.JobStatus.NEW.value, dataset_ids=dataset_ids))
        self.assertWarning()  
        return job1_id, owner, create_time, dataset_ids

    def test_job(self):
        job_id, owner, create_time, dataset_ids = self._add_job()
        job_id = 100000000
        self.db.update_job(job_id, constants.JobStatus.RUNNING.value)
        job = self.db.get_job(job_id)
        self.assertEqual(job.job_id, job_id)
        self.assertEqual(job.owner, owner)
        self.assertEqual(job.create_time, create_time)
        self.assertListEqual(list(job.dataset_ids), dataset_ids)


if __name__ == '__main__':
    unittest.main()