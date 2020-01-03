import os
import copy
import torch
import shutil
import logging
import requests
import importlib
import torch.nn.functional as F
from pfl.entity import runtime_config
from pfl.core.strategy import RunTimeStrategy
from pfl.utils.utils import LoggerFactory

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
LOCAL_MODEL_BASE_PATH = os.path.join(os.path.abspath("."), "res", "models")
AGGREGATE_PATH = "tmp_aggregate_pars"
THRESHOLD = 0.5


class TrainStrategy(object):
    def __init__(self, client_id):
        self.client_id = client_id
        self.fed_step = {}
        self.job_iter_dict = {}
        self.job_path = JOB_PATH
        self.logger = LoggerFactory.getLogger("TrainStrategy", logging.INFO)

    def _parse_optimizer(self, optimizer, model, lr):
        if optimizer == RunTimeStrategy.OPTIM_SGD.value:
            return torch.optim.SGD(model.parameters(), lr, momentum=0.5)

    def _compute_loss(self, loss_function, output, label):
        if loss_function == RunTimeStrategy.NLL_LOSS.value:
            loss = F.nll_loss(output, label)
        elif loss_function == RunTimeStrategy.KLDIV_LOSS.value:
            loss = F.kl_div(torch.log(output), label)
        return loss

    def _create_job_models_dir(self, client_id, job_id):
        # create local model dir
        local_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id))
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
        return local_model_dir

    def _load_job_model(self, job_id, job_model_class_name):
        # job_model_path = os.path.abspath(".") + "\\" + LOCAL_MODEL_BASE_PATH + "models_{}\\{}_init_model.py".format(
        #     job_id, job_id)
        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id),
                                         "init_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()

    def _find_latest_aggregate_model_pars(self, job_id):
        job_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "{}".format(AGGREGATE_PATH))
        if not os.path.exists(job_model_path):
            os.makedirs(job_model_path)
            init_model_pars_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id),
                                                "init_model_pars_{}".format(job_id))
            first_aggregate_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "tmp_aggregate_pars",
                                                "avg_pars_{}".format(0))
            if os.path.exists(init_model_pars_path):
                shutil.move(init_model_pars_path, first_aggregate_path)
        file_list = os.listdir(job_model_path)

        if len(file_list) != 0:
            return os.path.join(job_model_path, file_list[-1]), len(file_list)
        return None, 0


class TrainNormalStrategy(TrainStrategy):
    def __init__(self, job, data, fed_step, client_id):
        super(TrainNormalStrategy, self).__init__(client_id)
        self.job = job
        self.data = data
        self.job_model_path = os.path.join(os.path.abspath("."), "models_{}".format(job.get_job_id()))
        self.fed_step = fed_step

    def train(self):
        pass

    def start(self):
        pass

    def _train(self, train_model, job_models_path, fed_step):
        train_strategy = self.job.get_train_strategy()
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=train_strategy.get_batch_size(), shuffle=True,
                                                 num_workers=1,
                                                 pin_memory=True)

        optimizer = self._parse_optimizer(train_strategy.get_optimizer(), train_model,
                                          train_strategy.get_learning_rate())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        acc = 0
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            train_model = train_model.to(device)
            pred = train_model(batch_data)
            log_pred = torch.log(pred)
            loss = self._compute_loss(train_strategy.get_loss_function(), log_pred, batch_target)
            acc += torch.eq(pred.argmax(dim=1), batch_target).sum().float().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 200 == 0:
                self.logger.info("train_loss: {}".format(loss.item()))
                # print("loss: ", loss.item())

        torch.save(train_model.state_dict(),
                   os.path.join(job_models_path, "tmp_parameters_{}".format(fed_step)))
        return acc / len(dataloader.dataset)

    def _exec_finish_job(self, job_list):
        pass

    def _prepare_jobs_model(self, job_list):
        for job in job_list:
            self._prepare_job_model(job)

    def _prepare_job_model(self, job):
        # prepare job model py file
        job_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job.get_job_id()))
        job_init_model_path = os.path.join(job_model_path, "init_model_{}.py".format(job.get_job_id()))
        with open(job.get_train_model(), "r") as model_f:
            if not os.path.exists(job_init_model_path):
                f = open(job_init_model_path, "w")
                for line in model_f.readlines():
                    f.write(line)
                f.close()

    def _prepare_job_init_model_pars(self, job, server_url):
        job_init_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH,
                                               "models_{}".format(job.get_job_id()), "tmp_aggregate_pars")
        if not os.path.exists(job_init_model_pars_dir):
            os.makedirs(job_init_model_pars_dir)
        if len(os.listdir(job_init_model_pars_dir)) == 0:
            # print("/".join([server_url, "modelpars", job.get_job_id()]))
            response = requests.get("/".join([server_url, "modelpars", job.get_job_id()]))
            self._write_bfile_to_local(response, os.path.join(job_init_model_pars_dir, "avg_pars_0"))

    def _write_bfile_to_local(self, response, path):
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)

    def _prepare_upload_client_model_pars(self, job_id, client_id, fed_avg):
        job_init_model_pars_dir = os.path.join(os.path.abspath("."), LOCAL_MODEL_BASE_PATH,
                                               "models_{}".format(job_id), "models_{}".format(client_id))
        tmp_parameter_path = "tmp_parameters_{}".format(fed_avg)

        files = {
            'tmp_parameter_file': (
                'tmp_parameter_file', open(os.path.join(job_init_model_pars_dir, tmp_parameter_path), "rb"))
        }
        return files

    def _save_final_parameters(self, job_id, final_pars_path):
        file_path = os.path.join(os.path.abspath("."), "final_model_pars_{}".format(job_id))
        if os.path.exists(file_path):
            return
        with open(file_path, "wb") as w_f:
            with open(final_pars_path, "rb") as r_f:
                for line in r_f.readlines():
                    w_f.write(line)


class TrainDistillationStrategy(TrainNormalStrategy):
    def __init__(self, job, data, fed_step, client_id):
        super(TrainDistillationStrategy, self).__init__(job, data, fed_step, client_id)
        self.job_model_path = os.path.join(os.path.abspath("."), "res", "models", "models_{}".format(job.get_job_id()))

    def _load_other_models_pars(self, job_id, fed_step):
        job_model_base_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id))
        other_models_pars = []
        fed_step += 1
        connected_clients_num = 0
        for f in os.listdir(job_model_base_path):
            if f.find("models_") != -1:
                connected_clients_num += 1
                files = os.listdir(os.path.join(job_model_base_path, f))
                if len(files) == 0 or int(files[-1].split("_")[-1]) < fed_step:
                    return other_models_pars, 0
                else:
                    other_models_pars.append(torch.load(os.path.join(job_model_base_path, f, files[-1])))
        return other_models_pars, connected_clients_num

    def _calc_rate(self, received, total):
        if total == 0:
            return 0
        return received / total

    def _train_with_kl(self, train_model, other_models_pars, job_models_path):
        train_strategy = self.job.get_train_strategy()

        dataloader = torch.utils.data.DataLoader(self.data, batch_size=train_strategy.get_batch_size(), shuffle=True,
                                                 num_workers=1,
                                                 pin_memory=True)
        optimizer = self._parse_optimizer(train_strategy.get_optimizer(), train_model,
                                          train_strategy.get_learning_rate())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_model, other_model = train_model.to(device), copy.deepcopy(train_model)
        acc = 0
        for idx, (batch_data, batch_target) in enumerate(dataloader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            kl_pred = train_model(batch_data)
            pred = torch.log(kl_pred)
            acc += torch.eq(kl_pred.argmax(dim=1), batch_target).sum().float().item()
            loss_kl = self._compute_loss(RunTimeStrategy.KLDIV_LOSS.value, kl_pred, kl_pred)
            for other_model_pars in other_models_pars:
                other_model.load_state_dict(other_model_pars)
                other_model_kl_pred = other_model(batch_data).detach()
                loss_kl += self._compute_loss(RunTimeStrategy.KLDIV_LOSS.value, kl_pred, other_model_kl_pred)

            loss_s = self._compute_loss(train_strategy.get_loss_function(), pred, batch_target)
            loss = loss_s + self.job.get_distillation_alpha() * loss_kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                # print("distillation_loss: ", loss.item())
                self.logger.info("distillation_loss: {}".format(loss.item()))

        torch.save(train_model.state_dict(),
                   os.path.join(job_models_path, "tmp_parameters_{}".format(self.fed_step[self.job.get_job_id()] + 1)))
        return acc / len(dataloader.dataset)

class TrainStandloneNormalStrategy(TrainNormalStrategy):
    def  __init__(self, job, data, fed_step, client_id):
        super(TrainStandloneNormalStrategy, self).__init__(job, data, fed_step, client_id)

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_train_strategy().get_epoch():
                self.logger.info("job_{} completed".format(self.job.get_job_id()))
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_train_strategy().get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            aggregate_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            if aggregate_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
                if self.job.get_job_id() in runtime_config.EXEC_JOB_LIST:
                    runtime_config.EXEC_JOB_LIST.remove(self.job.get_job_id())
                self.fed_step[self.job.get_job_id()] = fed_step
            if self.job.get_job_id() not in runtime_config.EXEC_JOB_LIST:
                job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                if aggregate_file is not None:
                    self.logger.info("load {} parameters".format(aggregate_file))
                    job_model.load_state_dict(torch.load(aggregate_file))
                job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
                self.logger.info("job_{} is training, Aggregator strategy: {}".format(self.job.get_job_id(),
                                                                                      self.job.get_aggregate_strategy()))
                runtime_config.EXEC_JOB_LIST.append(self.job.get_job_id())
                self.acc = self._train(job_model, job_models_path, self.fed_step.get(self.job.get_job_id))
                self.logger.info("job_{} {}th train accuracy: {}".format(self.job.get_job_id(),
                                                                         self.fed_step.get(self.job.get_job_id()),
                                                                         self.acc))


class TrainStandloneDistillationStrategy(TrainDistillationStrategy):
    def __init__(self, job, data, fed_step, client_id):
        super(TrainStandloneDistillationStrategy, self).__init__(job, data, fed_step, client_id)
        self.train_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(
                self.job.get_job_id()) is None else self.fed_step.get(self.job.get_job_id())
            # print("test_iter_num: ", self.job_iter_dict[self.job.get_job_id()])
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_train_strategy().get_epoch():
                final_pars_path = os.path.join(self.job_model_path, "models_{}".format(self.client_id),
                                               "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id())))
                if os.path.exists(final_pars_path):
                    self._save_final_parameters(self.job.get_job_id(), final_pars_path)
                    self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            aggregate_file, _ = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            other_model_pars, connected_clients_num = self._load_other_models_pars(self.job.get_job_id(),
                                                                     self.fed_step[self.job.get_job_id()])
            job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())

            if other_model_pars is not None and connected_clients_num and self._calc_rate(len(other_model_pars),
                                                                connected_clients_num) >= THRESHOLD:
                self.logger.info("model distillating....")
                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1
                self.acc = self._train_with_kl(job_model, other_model_pars, job_models_path)
                self.logger.info("model distillation success")
            else:
                init_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                                    "models_{}".format(self.client_id))
                if not os.path.exists(os.path.join(init_model_pars_dir, "tmp_parameters_{}".format(1))):
                    job_model.load_state_dict(torch.load(aggregate_file))
                    self._train(job_model, init_model_pars_dir, 1)



class TrainMPCNormalStrategy(TrainNormalStrategy):
    def __init__(self, job, data, fed_step, client_ip, client_port, server_url, client_id):
        super(TrainMPCNormalStrategy, self).__init__(job, data, fed_step, client_id)
        self.server_url = server_url
        self.client_ip = client_ip
        self.client_port = client_port

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            # print("test_iter_num: ", self.job_iter_dict[self.job.get_job_id()])
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_train_strategy().get_epoch():
                self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_train_strategy().get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            self._prepare_job_model(self.job)
            self._prepare_job_init_model_pars(self.job, self.server_url)
            aggregate_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            if aggregate_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
                job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
                job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                self.logger.info("load {} parameters".format(aggregate_file))
                job_model.load_state_dict(torch.load(aggregate_file))
                self.fed_step[self.job.get_job_id()] = fed_step
                self.acc = self._train(job_model, job_models_path, self.fed_step.get(self.job.get_job_id()))
                files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                               self.fed_step.get(self.job.get_job_id()))
                response = requests.post("/".join(
                    [self.server_url, "modelpars", "%s" % self.client_id, "%s" % self.job.get_job_id(),
                     "%s" % self.fed_step[self.job.get_job_id()]]),
                    data=None, files=files)
                # print(response)


class TrainMPCDistillationStrategy(TrainDistillationStrategy):
    def __init__(self, job, data, fed_step, client_ip, client_port, server_url, client_id):
        super(TrainMPCDistillationStrategy, self).__init__(job, data, fed_step, client_id)
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_url = server_url

    def train(self):
        while True:
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_train_strategy().get_epoch():
                final_pars_path = os.path.join(self.job_model_path, "models_{}".format(self.client_id),
                                               "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id()) + 1))
                if os.path.exists(final_pars_path):
                    self._save_final_parameters(self.job.get_job_id(), final_pars_path)
                    self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            self._prepare_job_model(self.job)
            self._prepare_job_init_model_pars(self.job, self.server_url)
            job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
            job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            response = requests.get("/".join([self.server_url, "otherclients", self.job.get_job_id()]))
            connected_clients_id = response.json()['data']
            for client_id in connected_clients_id:
                self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(
                    self.job.get_job_id()) is None else self.fed_step.get(self.job.get_job_id())
                response = requests.get("/".join(
                    [self.server_url, "otherparameters", '%s' % self.job.get_job_id(), '%s' % client_id,
                     '%s' % (self.fed_step.get(self.job.get_job_id()) + 1)]))
                parameter_path = os.path.join(job_models_path,
                                              "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id()) + 1))
                if response.status_code == 202:
                    self._write_bfile_to_local(response, parameter_path)
            other_model_pars, _ = self._load_other_models_pars(self.job.get_job_id(),
                                                               self.fed_step.get(self.job.get_job_id()))
            if other_model_pars is not None and self._calc_rate(len(other_model_pars),
                                                                len(connected_clients_id)) >= THRESHOLD:
                self.logger.info("model distillating....")
                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1
                self.acc = self._train_with_kl(job_model, other_model_pars,
                                    os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                                 "models_{}".format(self.client_id)))
                self.logger.info("model distillation success")
                files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                               self.fed_step.get(self.job.get_job_id()) + 1)
                response = requests.post("/".join(
                    [self.server_url, "modelpars", "%s" % self.client_id, self.job.get_job_id(),
                     "%s" % (self.fed_step.get(self.job.get_job_id()) + 1)]), data=None, files=files)
            else:
                job_model_client_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                                     "models_{}".format(self.client_id))
                if not os.path.exists(os.path.join(job_model_client_path, "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id()) + 1))):
                    self._train(job_model, job_model_client_path, self.fed_step.get(self.job.get_job_id()) + 1)
                    files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                                   self.fed_step.get(self.job.get_job_id()) + 1)
                    response = requests.post("/".join(
                        [self.server_url, "modelpars", "%s" % self.client_id, self.job.get_job_id(),
                         "%s" % (self.fed_step.get(self.job.get_job_id()) + 1)]), data=None, files=files)
