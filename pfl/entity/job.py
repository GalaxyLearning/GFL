class Job(object):

    def __init__(self, server_host, job_id, train_strategy, train_model, train_model_class_name, aggregate_strategy,
                 distillation_alpha=None):
        self.server_host = server_host
        self.job_id = job_id
        self.train_strategy = train_strategy
        self.train_model = train_model
        self.train_model_class_name = train_model_class_name
        self.aggregate_strategy = aggregate_strategy
        self.alpha = distillation_alpha

    def set_server_host(self, server_host):
        self.server_host = server_host

    def set_job_id(self, job_id):
        self.job_id = job_id

    def get_job_id(self):
        return self.job_id

    def set_train_strategy(self, train_strategy):
        self.train_strategy = train_strategy

    def set_train_model(self, train_model):
        self.train_model = train_model

    def set_train_model_class_name(self, train_model_class_name):
        self.train_model_class_name = train_model_class_name

    def get_train_model_class_name(self):
        return self.train_model_class_name

    def get_server_host(self):
        return self.server_host

    def get_train_strategy(self):
        return self.train_strategy

    def get_train_model(self):
        return self.train_model

    def set_aggregate_stragety(self, aggregate_strategy):
        self.aggregate_strategy = aggregate_strategy

    def get_aggregate_strategy(self):
        return self.aggregate_strategy

    def set_distillation_alpha(self, alpha):
        self.alpha = alpha

    def get_distillation_alpha(self):
        return self.alpha
