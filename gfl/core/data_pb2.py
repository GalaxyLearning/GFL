# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gfl/core/data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13gfl/core/data.proto\x12\x08gfl.core\"b\n\x07JobMeta\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\r\n\x05owner\x18\x02 \x01(\t\x12\x13\n\x0b\x63reate_time\x18\x03 \x01(\x03\x12\x0e\n\x06status\x18\x04 \x01(\x05\x12\x13\n\x0b\x64\x61taset_ids\x18\x05 \x03(\t\"0\n\tJobConfig\x12\x0f\n\x07trainer\x18\x01 \x01(\t\x12\x12\n\naggregator\x18\x02 \x01(\t\"{\n\x0bTrainConfig\x12\r\n\x05model\x18\x01 \x01(\t\x12\x11\n\toptimizer\x18\x02 \x01(\t\x12\x11\n\tcriterion\x18\x03 \x01(\t\x12\x14\n\x0clr_scheduler\x18\x04 \x01(\t\x12\r\n\x05\x65poch\x18\x05 \x01(\x05\x12\x12\n\nbatch_size\x18\x06 \x01(\x05\"\'\n\x0f\x41ggregateConfig\x12\x14\n\x0cglobal_epoch\x18\x01 \x01(\x05\"\xc6\x01\n\x07JobData\x12\x1f\n\x04meta\x18\x01 \x01(\x0b\x32\x11.gfl.core.JobMeta\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\t\x12\'\n\njob_config\x18\x03 \x01(\x0b\x32\x13.gfl.core.JobConfig\x12+\n\x0ctrain_config\x18\x04 \x01(\x0b\x32\x15.gfl.core.TrainConfig\x12\x33\n\x10\x61ggregate_config\x18\x05 \x01(\x0b\x32\x19.gfl.core.AggregateConfig\"\xfc\x01\n\x08JobTrace\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x17\n\x0f\x62\x65gin_timepoint\x18\x02 \x01(\x03\x12\x15\n\rend_timepoint\x18\x03 \x01(\x03\x12\x12\n\nready_time\x18\x04 \x01(\x05\x12\x1e\n\x16\x61ggregate_running_time\x18\x05 \x01(\x05\x12\x1e\n\x16\x61ggregate_waiting_time\x18\x06 \x01(\x05\x12\x1a\n\x12train_running_time\x18\x07 \x01(\x05\x12\x1a\n\x12train_waiting_time\x18\x08 \x01(\x05\x12\x11\n\tcomm_time\x18\t \x01(\x05\x12\x11\n\tused_time\x18\n \x01(\x05\"\x98\x01\n\x0b\x44\x61tasetMeta\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\r\n\x05owner\x18\x02 \x01(\t\x12\x13\n\x0b\x63reate_time\x18\x03 \x01(\x03\x12\x0e\n\x06status\x18\x04 \x01(\x05\x12\x0c\n\x04type\x18\x05 \x01(\x05\x12\x0c\n\x04size\x18\x06 \x01(\x05\x12\x13\n\x0brequest_cnt\x18\x07 \x01(\x05\x12\x10\n\x08used_cnt\x18\x08 \x01(\x05\"G\n\rDatasetConfig\x12\x0f\n\x07\x64\x61taset\x18\x01 \x01(\t\x12\x13\n\x0bval_dataset\x18\x02 \x01(\t\x12\x10\n\x08val_rate\x18\x03 \x01(\x02\"t\n\x0b\x44\x61tasetData\x12#\n\x04meta\x18\x01 \x01(\x0b\x32\x15.gfl.core.DatasetMeta\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\t\x12/\n\x0e\x64\x61taset_config\x18\x03 \x01(\x0b\x32\x17.gfl.core.DatasetConfig\"T\n\x0c\x44\x61tasetTrace\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x0e\n\x06job_id\x18\x02 \x01(\t\x12\x11\n\tconfirmed\x18\x03 \x01(\x08\x12\r\n\x05score\x18\x04 \x01(\x02\"\xcf\x01\n\x0bModelParams\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x14\n\x0cnode_address\x18\x02 \x01(\t\x12\x12\n\ndataset_id\x18\x03 \x01(\t\x12\x0c\n\x04step\x18\x04 \x01(\x05\x12\x0c\n\x04path\x18\x05 \x01(\t\x12\x0c\n\x04loss\x18\x06 \x01(\x02\x12\x13\n\x0bmetric_name\x18\x07 \x01(\t\x12\x14\n\x0cmetric_value\x18\x08 \x01(\x02\x12\r\n\x05score\x18\t \x01(\x02\x12\x14\n\x0cis_aggregate\x18\n \x01(\x08\x12\x0c\n\x04\x64\x61ta\x18\x0b \x01(\x0c\",\n\x08NodeInfo\x12\x0f\n\x07\x61\x64\x64ress\x18\x01 \x01(\t\x12\x0f\n\x07pub_key\x18\x02 \x01(\tb\x06proto3')



_JOBMETA = DESCRIPTOR.message_types_by_name['JobMeta']
_JOBCONFIG = DESCRIPTOR.message_types_by_name['JobConfig']
_TRAINCONFIG = DESCRIPTOR.message_types_by_name['TrainConfig']
_AGGREGATECONFIG = DESCRIPTOR.message_types_by_name['AggregateConfig']
_JOBDATA = DESCRIPTOR.message_types_by_name['JobData']
_JOBTRACE = DESCRIPTOR.message_types_by_name['JobTrace']
_DATASETMETA = DESCRIPTOR.message_types_by_name['DatasetMeta']
_DATASETCONFIG = DESCRIPTOR.message_types_by_name['DatasetConfig']
_DATASETDATA = DESCRIPTOR.message_types_by_name['DatasetData']
_DATASETTRACE = DESCRIPTOR.message_types_by_name['DatasetTrace']
_MODELPARAMS = DESCRIPTOR.message_types_by_name['ModelParams']
_NODEINFO = DESCRIPTOR.message_types_by_name['NodeInfo']
JobMeta = _reflection.GeneratedProtocolMessageType('JobMeta', (_message.Message,), {
  'DESCRIPTOR' : _JOBMETA,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.JobMeta)
  })
_sym_db.RegisterMessage(JobMeta)

JobConfig = _reflection.GeneratedProtocolMessageType('JobConfig', (_message.Message,), {
  'DESCRIPTOR' : _JOBCONFIG,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.JobConfig)
  })
_sym_db.RegisterMessage(JobConfig)

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), {
  'DESCRIPTOR' : _TRAINCONFIG,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.TrainConfig)
  })
_sym_db.RegisterMessage(TrainConfig)

AggregateConfig = _reflection.GeneratedProtocolMessageType('AggregateConfig', (_message.Message,), {
  'DESCRIPTOR' : _AGGREGATECONFIG,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.AggregateConfig)
  })
_sym_db.RegisterMessage(AggregateConfig)

JobData = _reflection.GeneratedProtocolMessageType('JobData', (_message.Message,), {
  'DESCRIPTOR' : _JOBDATA,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.JobData)
  })
_sym_db.RegisterMessage(JobData)

JobTrace = _reflection.GeneratedProtocolMessageType('JobTrace', (_message.Message,), {
  'DESCRIPTOR' : _JOBTRACE,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.JobTrace)
  })
_sym_db.RegisterMessage(JobTrace)

DatasetMeta = _reflection.GeneratedProtocolMessageType('DatasetMeta', (_message.Message,), {
  'DESCRIPTOR' : _DATASETMETA,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.DatasetMeta)
  })
_sym_db.RegisterMessage(DatasetMeta)

DatasetConfig = _reflection.GeneratedProtocolMessageType('DatasetConfig', (_message.Message,), {
  'DESCRIPTOR' : _DATASETCONFIG,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.DatasetConfig)
  })
_sym_db.RegisterMessage(DatasetConfig)

DatasetData = _reflection.GeneratedProtocolMessageType('DatasetData', (_message.Message,), {
  'DESCRIPTOR' : _DATASETDATA,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.DatasetData)
  })
_sym_db.RegisterMessage(DatasetData)

DatasetTrace = _reflection.GeneratedProtocolMessageType('DatasetTrace', (_message.Message,), {
  'DESCRIPTOR' : _DATASETTRACE,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.DatasetTrace)
  })
_sym_db.RegisterMessage(DatasetTrace)

ModelParams = _reflection.GeneratedProtocolMessageType('ModelParams', (_message.Message,), {
  'DESCRIPTOR' : _MODELPARAMS,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.ModelParams)
  })
_sym_db.RegisterMessage(ModelParams)

NodeInfo = _reflection.GeneratedProtocolMessageType('NodeInfo', (_message.Message,), {
  'DESCRIPTOR' : _NODEINFO,
  '__module__' : 'gfl.core.data_pb2'
  # @@protoc_insertion_point(class_scope:gfl.core.NodeInfo)
  })
_sym_db.RegisterMessage(NodeInfo)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _JOBMETA._serialized_start=33
  _JOBMETA._serialized_end=131
  _JOBCONFIG._serialized_start=133
  _JOBCONFIG._serialized_end=181
  _TRAINCONFIG._serialized_start=183
  _TRAINCONFIG._serialized_end=306
  _AGGREGATECONFIG._serialized_start=308
  _AGGREGATECONFIG._serialized_end=347
  _JOBDATA._serialized_start=350
  _JOBDATA._serialized_end=548
  _JOBTRACE._serialized_start=551
  _JOBTRACE._serialized_end=803
  _DATASETMETA._serialized_start=806
  _DATASETMETA._serialized_end=958
  _DATASETCONFIG._serialized_start=960
  _DATASETCONFIG._serialized_end=1031
  _DATASETDATA._serialized_start=1033
  _DATASETDATA._serialized_end=1149
  _DATASETTRACE._serialized_start=1151
  _DATASETTRACE._serialized_end=1235
  _MODELPARAMS._serialized_start=1238
  _MODELPARAMS._serialized_end=1445
  _NODEINFO._serialized_start=1447
  _NODEINFO._serialized_end=1491
# @@protoc_insertion_point(module_scope)
