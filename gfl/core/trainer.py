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

import os
import os
import copy
import torch
import shutil
import logging
import requests
import importlib
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gfl.entity import runtime_config
from gfl.exceptions.fl_expection import GFLException
from gfl.core.strategy import OptimizerStrategy, LossStrategy, SchedulerStrategy
from gfl.utils.utils import LoggerFactory

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
LOCAL_MODEL_BASE_PATH = os.path.join(os.path.abspath("."), "res", "models")
AGGREGATE_PATH = "tmp_aggregate_pars"
THRESHOLD = 0.5


class TrainStrategy(object):
    """
    TrainStrategy is the root class of all train strategy classes
    """

    def __init__(self, client_id):
        self.client_id = client_id
        self.fed_step = {}
        self.job_iter_dict = {}
        self.job_path = JOB_PATH

    def _parse_optimizer(self, optimizer, model, lr):
        if optimizer == OptimizerStrategy.OPTIM_SGD.value:
            return torch.optim.SGD(model.parameters(), lr, momentum=0.5)

    def _compute_loss(self, loss_function, output, label):
        """
        Return the loss according to the loss_function
        :param loss_function:
        :param output:
        :param label:
        :return:
        """
        if loss_function == LossStrategy.NLL_LOSS:
            loss = F.nll_loss(output, label)
        elif loss_function == LossStrategy.KLDIV_LOSS:
            loss = F.kl_div(torch.log(output), label, reduction='batchmean')
        return loss

    def _compute_l2_dist(self, output, label):
        loss = F.mse_loss(output, label)
        return loss

    def _create_job_models_dir(self, client_id, job_id):
        """
        Create local temporary model directory according to client_id and job_id
        :param client_id:
        :param job_id:
        :return:
        """
        local_model_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job_id), "models_{}".format(client_id))
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
        return local_model_dir

    def _load_job_model(self, job_id, job_model_class_name):
        """
        Load model object according to job_id and model's class name
        :param job_id:
        :param job_model_class_name:
        :return:
        """
        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id),
                                         "init_model_{}".format(job_id))
        model_class = getattr(module, job_model_class_name)
        return model_class()

    def _find_latest_aggregate_model_pars(self, job_id):
        """
        Return the latest aggregated model's parameters
        :param job_id:
        :return:
        """
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
        file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(job_model_path, x)))
        
        if len(file_list) != 0:
            return os.path.join(job_model_path, file_list[-1]), len(file_list)
        return None, 0


class TrainNormalStrategy(TrainStrategy):
    """
    TrainNormalStrategy provides traditional training method and some necessary methods
    """

    def __init__(self, job, data, fed_step, client_id, local_epoch, model, curve):
        super(TrainNormalStrategy, self).__init__(client_id)
        self.job = job
        self.data = data
        self.job_model_path = os.path.join(os.path.abspath("."), "models_{}".format(job.get_job_id()))
        self.fed_step = fed_step
        self.local_epoch = local_epoch
        self.accuracy_list = []
        self.loss_list = []
        self.model = model
        self.curve = curve

    def train(self):
        pass

    def _train(self, train_model, job_models_path, fed_step, local_epoch):
        """
        Traditional training method
        :param train_model:
        :param job_models_path:
        :param fed_step:
        :return:
        """
        # TODO: transfer training code to c++ and invoked by python using pybind11

        step = 0
        model = train_model.get_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        while step < local_epoch:
            dataloader = torch.utils.data.DataLoader(self.data,
                                                     batch_size=train_model.get_train_strategy().get_batch_size(),
                                                     shuffle=True,
                                                     num_workers=1,
                                                     pin_memory=True)


            acc = 0

            if train_model.get_train_strategy().get_optimizer() is not None:
                optimizer = self._generate_new_optimizer(model, train_model.get_train_strategy().get_optimizer())
            else:
                optimizer = self._generate_new_scheduler(model, train_model.get_train_strategy().get_scheduler())
            for idx, (batch_data, batch_target) in enumerate(dataloader):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                pred = model(batch_data)
                log_pred = torch.log(F.softmax(pred, dim=1))
                loss = self._compute_loss(train_model.get_train_strategy().get_loss_function(), log_pred, batch_target)
                acc += torch.eq(pred.argmax(dim=1), batch_target).sum().float().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx % 200 == 0:
                    self.logger.info("train_loss: {}".format(loss.item()))
            step += 1
            accuracy = acc / len(dataloader.dataset)
        torch.save(model.state_dict(),
                       os.path.join(job_models_path, "tmp_parameters_{}".format(fed_step)))

        return accuracy, loss.item()

    def _exec_finish_job(self, job_list):
        pass

    def _prepare_jobs_model(self, job_list):
        for job in job_list:
            self._prepare_job_model(job, None)

    def _prepare_job_model(self, job, server_url=None):
        job_model_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(job.get_job_id()))
        job_init_model_path = os.path.join(job_model_path, "init_model_{}.py".format(job.get_job_id()))
        if server_url is None:
            # with open(job.get_train_model(), "r") as model_f:
            #     if not os.path.exists(job_init_model_path):
            #         f = open(job_init_model_path, "w")
            #         for line in model_f.readlines():
            #             f.write(line)
            #         f.close()
            if not os.path.exists(job_init_model_path):
                with open(job_init_model_path, "w") as model_f:
                    with open(job.get_train_model(), "r") as f:
                        for line in f.readlines():
                            model_f.write(line)
        else:
            if not os.path.exists(job_model_path):
                os.makedirs(job_model_path)
            if not os.path.exists(job_init_model_path):
                response = requests.get("/".join([server_url, "init_model", job.get_job_id()]))
                self._write_bfile_to_local(response, job_init_model_path)

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

    def _generate_new_optimizer(self, model, optimizer):
        state_dict = optimizer.state_dict()
        optimizer_class = optimizer.__class__
        params = state_dict['param_groups'][0]
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise GFLException("optimizer get wrong type value")

        if isinstance(optimizer, torch.optim.SGD):
            return optimizer_class(model.parameters(), lr=params['lr'], momentum=params['momentum'],
                                   dampening=params['dampening'], weight_decay=params['weight_decay'],
                                   nesterov=params['nesterov'])
        else:
            return optimizer_class(model.parameters(), lr=params['lr'], betas=params['betas'],
                                   eps=params['eps'], weight_decay=params['weight_decay'],
                                   amsgrad=params['amsgrad'])

    def _generate_new_scheduler(self, model, scheduler):
        scheduler_names = []
        for scheduler_item in SchedulerStrategy.__members__.items():
            scheduler_names.append(scheduler_item.value)
        if scheduler.__class__.__name__ not in scheduler_names:
            raise GFLException("optimizer get wrong type value")
        optimizer = scheduler.__getattribute__("optimizer")
        params = scheduler.state_dict()
        new_optimizer = self._generate_new_optimizer(model, optimizer)
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            return torch.optim.lr_scheduler.CyclicLR(new_optimizer, base_lr=params['base_lrs'],
                                                     max_lr=params['max_lrs'],
                                                     step_size_up=params['total_size'] * params['step_ratio'],
                                                     step_size_down=params['total_size'] - (params['total_size'] *
                                                                                            params['step_ratio']),
                                                     mode=params['mode'], gamma=params['gamma'],
                                                     scale_fn=params['scale_fn'], scale_mode=params['scale_mode'],
                                                     cycle_momentum=params['cycle_momentum'],
                                                     base_momentum=params['base_momentums'],
                                                     max_momentum=params['max_momentums'],
                                                     last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                         'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            return torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=params['T_max'],
                                                              eta_min=params['eta_min'],
                                                              last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                                  'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
            return torch.optim.lr_scheduler.ExponentialLR(new_optimizer, gamma=params['gamma'],
                                                          last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                              'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            return torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=params['lr_lamdas'],
                                                     last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                         'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
            return torch.optim.lr_scheduler.MultiStepLR(new_optimizer, milestones=params['milestones'],
                                                        gamma=params['gammas'],
                                                        last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                            'last_epoch']))
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(new_optimizer, mode=params['mode'],
                                                              factor=params['factor'], patience=params['patience'],
                                                              verbose=params['verbose'], threshold=params['threshold'],
                                                              threshold_mode=params['threshold_mode'],
                                                              cooldown=params['cooldown'], min_lr=params['min_lrs'],
                                                              eps=params['eps'])
        elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            return torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=params['step_size'], gamma=params['gamma'],
                                                   last_epoch=(-1 if params['last_epoch'] == 0 else params[
                                                       'last_epoch']))

    def _draw_curve(self):
        loss_x = range(0, self.job.get_epoch())
        accuracy_x = range(0, self.job.get_epoch())
        loss_y = self.loss_list
        accuracy_y = self.accuracy_list
        plt.subplot(2, 1, 1)
        plt.plot(loss_x, loss_y, ".-")
        plt.title("Train loss curve")
        plt.ylabel("Train loss")
        plt.xlabel("epoch")

        plt.subplot(2, 1, 2)
        plt.plot(accuracy_x, accuracy_y, "o-")
        plt.title("Train accuracy curve")
        plt.ylabel("Train accuracy")
        plt.xlabel("epoch")
        plt.show()


class TrainDistillationStrategy(TrainNormalStrategy):
    """
    TrainDistillationStrategy provides distillation training method and some necessary methods
    """

    def __init__(self, job, data, fed_step, client_id, models, curve):
        super(TrainDistillationStrategy, self).__init__(job, data, fed_step, client_id, models, curve)
        self.job_model_path = os.path.join(os.path.abspath("."), "res", "models", "models_{}".format(job.get_job_id()))

    def _load_other_models_pars(self, job_id, fed_step):
        """
        Load model's pars from other clients in fed_step round
        :param job_id:
        :param fed_step:
        :return:
        """
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
        """
        Calculate response rate of clients
        :param received:
        :param total:
        :return:
        """
        if total == 0:
            return 0
        return received / total

    def _train_with_distillation(self, train_model, other_models_pars, local_epoch, job_models_path, job_l2_dist):
        """
        Distillation training method
        :param train_model:
        :param other_models_pars:
        :param job_models_path:
        :return:
        """
        # TODO: transfer training code to c++ and invoked by python using pybind11
        step = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train_model.get_model()
        model, other_model = model.to(device), copy.deepcopy(model).to(device)
        while step < local_epoch:
            dataloader = torch.utils.data.DataLoader(self.data,
                                                     batch_size=train_model.get_train_strategy().get_batch_size(),
                                                     shuffle=True,
                                                     num_workers=1,
                                                     pin_memory=True)

            optimizer = self._generate_new_optimizer(model, train_model.get_train_strategy().get_optimizer())
            acc = 0
            for idx, (batch_data, batch_target) in enumerate(dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                kl_pred = model(batch_data)
                pred = torch.log(F.softmax(kl_pred, dim=1))
                acc += torch.eq(kl_pred.argmax(dim=1), batch_target).sum().float().item()
                if job_l2_dist:
                    loss_distillation = self._compute_l2_dist(kl_pred, kl_pred)
                else:
                    loss_distillation = self._compute_loss(LossStrategy.KLDIV_LOSS, F.softmax(kl_pred, dim=1),
                                                           F.softmax(kl_pred, dim=1))
                for other_model_pars in other_models_pars:
                    other_model.load_state_dict(other_model_pars)
                    other_model_kl_pred = other_model(batch_data).detach()
                    if job_l2_dist:
                        loss_distillation += self._compute_l2_dist(kl_pred, other_model_kl_pred)
                    else:
                        loss_distillation += self._compute_loss(LossStrategy.KLDIV_LOSS, F.softmax(kl_pred, dim=1),
                                                                F.softmax(other_model_kl_pred, dim=1))

                loss_s = self._compute_loss(train_model.get_train_strategy().get_loss_function(), pred, batch_target)
                loss = loss_s + self.job.get_distillation_alpha() * loss_distillation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 200 == 0:
                    # print("distillation_loss: ", loss.item())
                    self.logger.info("distillation_loss: {}".format(loss.item()))
            step += 1
            accuracy = acc / len(dataloader.dataset)
        torch.save(model.state_dict(),
                       os.path.join(job_models_path, "tmp_parameters_{}".format(self.fed_step[self.job.get_job_id()] + 1)))
        return accuracy, loss.item()


class TrainStandloneNormalStrategy(TrainNormalStrategy):
    """
    TrainStandloneNormalStrategy is responsible for controlling the process of traditional training in standalone mode
    """

    def __init__(self, job, data, fed_step, client_id, local_epoch, model, curve):
        super(TrainStandloneNormalStrategy, self).__init__(job, data, fed_step, client_id, local_epoch, model, curve)
        self.logger = LoggerFactory.getLogger("TrainStandloneNormalStrategy", logging.INFO)

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_epoch():
                self.logger.info("job_{} completed".format(self.job.get_job_id()))
                if self.curve is True:
                    self._draw_curve()
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                break
            aggregate_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            if aggregate_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
                if self.job.get_job_id() in runtime_config.EXEC_JOB_LIST:
                    runtime_config.EXEC_JOB_LIST.remove(self.job.get_job_id())
                self.fed_step[self.job.get_job_id()] = fed_step
            if self.job.get_job_id() not in runtime_config.EXEC_JOB_LIST:
                # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                if aggregate_file is not None:
                    self.logger.info("load {} parameters".format(aggregate_file))
                    new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                    model_pars = torch.load(aggregate_file)
                    new_model.load_state_dict(model_pars)
                    self.model.set_model(new_model)
                job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
                self.logger.info("job_{} is training, Aggregator strategy: {}".format(self.job.get_job_id(),
                                                                                      self.job.get_aggregate_strategy()))
                runtime_config.EXEC_JOB_LIST.append(self.job.get_job_id())
                self.acc, loss = self._train(self.model, job_models_path, self.fed_step.get(self.job.get_job_id()), self.local_epoch)
                self.loss_list.append(loss)
                self.accuracy_list.append(self.acc)
                self.logger.info("job_{} {}th train accuracy: {}".format(self.job.get_job_id(),
                                                                         self.fed_step.get(self.job.get_job_id()),
                                                                         self.acc))


class TrainStandloneDistillationStrategy(TrainDistillationStrategy):
    """
    TrainStandloneDistillationStrategy is responsible for controlling the process of distillation training in standalone mode
    """

    def __init__(self, job, data, fed_step, client_id, local_epoch, model, curve):
        super(TrainStandloneDistillationStrategy, self).__init__(job, data, fed_step, client_id, local_epoch, model, curve)
        self.train_model = self._load_job_model(job.get_job_id(), job.get_train_model_class_name())
        self.logger = LoggerFactory.getLogger("TrainStandloneDistillationStrategy", logging.INFO)

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(
                self.job.get_job_id()) is None else self.fed_step.get(self.job.get_job_id())
            # print("test_iter_num: ", self.job_iter_dict[self.job.get_job_id()])
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_epoch():
                final_pars_path = os.path.join(self.job_model_path, "models_{}".format(self.client_id),
                                               "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id())))
                if os.path.exists(final_pars_path):
                    self._save_final_parameters(self.job.get_job_id(), final_pars_path)
                    self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            aggregate_file, _ = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            other_model_pars, connected_clients_num = self._load_other_models_pars(self.job.get_job_id(),
                                                                                   self.fed_step[self.job.get_job_id()])
            # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
            job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())

            self.logger.info("job_{} is training, Aggregator strategy: {}, L2_dist: {}".format(self.job.get_job_id(),
                                                                                               self.job.get_aggregate_strategy(),
                                                                                               self.job.get_l2_dist()))
            if other_model_pars is not None and connected_clients_num and self._calc_rate(len(other_model_pars),
                                                                                          connected_clients_num) >= THRESHOLD:

                self.logger.info("model distillating....")
                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1
                self.acc, loss = self._train_with_distillation(self.model, other_model_pars, self.local_epoch, job_models_path,
                                                               self.job.get_l2_dist())
                self.accuracy_list.append(self.acc)
                self.loss_list.append(loss)
                self.logger.info("model distillation success")
            else:
                init_model_pars_dir = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                                   "models_{}".format(self.client_id))
                if not os.path.exists(os.path.join(init_model_pars_dir, "tmp_parameters_{}".format(1))):
                    new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                    model_pars = torch.load(aggregate_file)
                    new_model.load_state_dict(model_pars)
                    self.model.set_model(new_model)
                    self._train(self.model, init_model_pars_dir, 1, self.local_epoch)


class TrainMPCNormalStrategy(TrainNormalStrategy):
    """
    TrainMPCNormalStrategy is responsible for controlling the process of traditional training in cluster mode
    """

    def __init__(self, job, data, fed_step, client_ip, client_port, server_url, client_id, local_epoch, model, curve):
        super(TrainMPCNormalStrategy, self).__init__(job, data, fed_step, client_id, local_epoch, model, curve)
        self.server_url = server_url
        self.client_ip = client_ip
        self.client_port = client_port
        self.logger = LoggerFactory.getLogger("TrainMPCNormalStrategy", logging.INFO)

    def train(self):
        while True:
            self.fed_step[self.job.get_job_id()] = 0 if self.fed_step.get(self.job.get_job_id()) is None else \
                self.fed_step.get(self.job.get_job_id())
            # print("test_iter_num: ", self.job_iter_dict[self.job.get_job_id()])
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) == self.job.get_epoch():
                self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            elif self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) > self.job.get_epoch():
                self.logger.warning("job_{} has completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            self._prepare_job_model(self.job, self.server_url)
            self._prepare_job_init_model_pars(self.job, self.server_url)
            aggregate_file, fed_step = self._find_latest_aggregate_model_pars(self.job.get_job_id())
            if aggregate_file is not None and self.fed_step.get(self.job.get_job_id()) != fed_step:
                job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
                # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                self.logger.info("load {} parameters".format(aggregate_file))
                new_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
                model_pars = torch.load(aggregate_file)
                new_model.load_state_dict(model_pars)
                self.model.set_model(new_model)
                self.fed_step[self.job.get_job_id()] = fed_step
                self.logger.info("job_{} is training, Aggregator strategy: {}".format(self.job.get_job_id(),
                                                                                      self.job.get_aggregate_strategy()))
                self.acc, loss = self._train(self.model, job_models_path, self.fed_step.get(self.job.get_job_id()), self.local_epoch)
                self.loss_list.append(loss)
                self.accuracy_list.append(self.acc)
                files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                               self.fed_step.get(self.job.get_job_id()))
                response = requests.post("/".join(
                    [self.server_url, "modelpars", "%s" % self.client_id, "%s" % self.job.get_job_id(),
                     "%s" % self.fed_step[self.job.get_job_id()]]),
                    data=None, files=files)
                # print(response)


class TrainMPCDistillationStrategy(TrainDistillationStrategy):
    """
    TrainMPCDistillationStrategy is responsible for controlling the process of distillation training in cluster mode
    """

    def __init__(self, job, data, fed_step, client_ip, client_port, server_url, client_id, local_epoch, model, curve):
        super(TrainMPCDistillationStrategy, self).__init__(job, data, fed_step, client_id, local_epoch, model, curve)
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_url = server_url
        self.logger = LoggerFactory.getLogger("TrainMPCDistillationStrategy", logging.INFO)

    def train(self):
        while True:
            if self.fed_step.get(self.job.get_job_id()) is not None and self.fed_step.get(
                    self.job.get_job_id()) >= self.job.get_epoch():
                final_pars_path = os.path.join(self.job_model_path, "models_{}".format(self.client_id),
                                               "tmp_parameters_{}".format(self.fed_step.get(self.job.get_job_id()) + 1))
                if os.path.exists(final_pars_path):
                    self._save_final_parameters(self.job.get_job_id(), final_pars_path)
                    self.logger.info("job_{} completed, final accuracy: {}".format(self.job.get_job_id(), self.acc))
                if self.curve is True:
                    self._draw_curve()
                break
            self._prepare_job_model(self.job)
            self._prepare_job_init_model_pars(self.job, self.server_url)
            job_models_path = self._create_job_models_dir(self.client_id, self.job.get_job_id())
            # job_model = self._load_job_model(self.job.get_job_id(), self.job.get_train_model_class_name())
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

            self.logger.info("job_{} is training, Aggregator strategy: {}, L2_dist: {}".format(self.job.get_job_id(),
                                                                                               self.job.get_aggregate_strategy(),
                                                                                               self.job.get_l2_dist()))
            if other_model_pars is not None and self._calc_rate(len(other_model_pars),
                                                                len(connected_clients_id)) >= THRESHOLD:

                self.logger.info("model distillating....")
                self.fed_step[self.job.get_job_id()] = self.fed_step.get(self.job.get_job_id()) + 1
                self.acc, loss = self._train_with_distillation(self.model, other_model_pars, self.local_epoch,
                                                               os.path.join(LOCAL_MODEL_BASE_PATH,
                                                                            "models_{}".format(self.job.get_job_id()),
                                                                            "models_{}".format(self.client_id)),
                                                               self.job.get_l2_dist())
                self.loss_list.append(loss)
                self.accuracy_list.append(self.acc)
                self.logger.info("model distillation success")
                files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                               self.fed_step.get(self.job.get_job_id()) + 1)
                response = requests.post("/".join(
                    [self.server_url, "modelpars", "%s" % self.client_id, self.job.get_job_id(),
                     "%s" % (self.fed_step.get(self.job.get_job_id()) + 1)]), data=None, files=files)
            else:
                job_model_client_path = os.path.join(LOCAL_MODEL_BASE_PATH, "models_{}".format(self.job.get_job_id()),
                                                     "models_{}".format(self.client_id))
                if not os.path.exists(os.path.join(job_model_client_path, "tmp_parameters_{}".format(
                        self.fed_step.get(self.job.get_job_id()) + 1))):
                    self._train(self.model, job_model_client_path, self.fed_step.get(self.job.get_job_id()) + 1, self.local_epoch)
                    files = self._prepare_upload_client_model_pars(self.job.get_job_id(), self.client_id,
                                                                   self.fed_step.get(self.job.get_job_id()) + 1)
                    response = requests.post("/".join(
                        [self.server_url, "modelpars", "%s" % self.client_id, self.job.get_job_id(),
                         "%s" % (self.fed_step.get(self.job.get_job_id()) + 1)]), data=None, files=files)
