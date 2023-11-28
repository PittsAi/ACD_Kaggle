import os
import re
import json
import argparse
import torch
import random
import numpy as np
from transformers import BertConfig

from types import SimpleNamespace as Namespace

def prepare_model_config(args):
    config = BertConfig.from_pretrained(args.model.model_name_or_path)
    config.model_name_or_path = args.model.model_name_or_path
    config.num_labels = len(args.data.label_scheme)
    config.output_attentions = args.model.output_attentions
    config.output_hidden_states = args.model.output_hidden_states
    config.visual_embedding_dim = args.data.image.embedding_dim
    config.visual_embedding_size = args.data.image.size
    config.ckpt_path = args.model.pretrained_weights

    return config


class Arguments(object):
    def __init__(self):
        self.args = self.get_args()
        self._read_json_params()
        self._format_datapaths()
        self._add_extra_fields()
        self.get_seed()
        self._print_args()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='./configs/ACD-baseline.json', help='Provide the JSON config path with the parameters of your experiment')
        parser.add_argument('--replicable', type=bool, default=True, help='')
        parser.add_argument('--is_cuda', type=bool, default=False, help='')
        parser.add_argument('--log_interval', type=int, default=10, help='')

        return parser.parse_args()


    def _format_datapaths(self):
        # self.data.directory = os.path.join(glb.PROJ_DIR, self.data.directory)

        self.data.text.train = os.path.join(self.data.directory, self.data.text.train)
        self.data.text.valid = os.path.join(self.data.directory, self.data.text.valid)
        self.data.text.test = os.path.join(self.data.directory, self.data.text.test)

        self.data.text.train_label = os.path.join(self.data.directory, self.data.label.train)
        self.data.text.valid_label = os.path.join(self.data.directory, self.data.label.valid)
        self.data.text.test_label = os.path.join(self.data.directory, self.data.label.test)

    def _add_extra_fields(self):
        # self.experiment.output_dir = os.path.join(glb.PROJ_DIR, self.experiment.output_dir, self.experiment.id)
        self.experiment.output_dir = self.experiment.output_dir, self.experiment.id

        self.experiment.checkpoint_dir = os.path.join(self.experiment.output_dir[0],self.experiment.output_dir[1], 'checkpoint')

    def _read_json_params(self):
        # Read the parameters from the JSON file and skip comments
        with open(self.args.config, 'r') as f:
            params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])
        
        arguments = json.loads(params, object_hook=lambda d: Namespace(**d))

        # with open(self.args.config, 'r') as f:
        #     arguments = json.load(f, object_hook=lambda d: Namespace(**d))
        # print(arguments)

        # Must-have fields expected from the JSON config file
        self.experiment = arguments.experiment
        self.data = arguments.data
        self.model = arguments.model
        self.training = arguments.training
        self.optim = self.training.optim

        # Optim Args
        self.optim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Checking that the JSON contains at least the fixed fields
        assert all([hasattr(self.data.text, name) for name in {'train', 'valid', 'test'}])
        assert all([hasattr(self.training, name) for name in
                    {'epochs', 'per_gpu_train_batch_size', 'per_gpu_eval_batch_size', 'optim'}])
        assert all([hasattr(self.training.optim, name) for name in {'learning_rate', 'weight_decay'}])

    def get_seed(self):
        # Fields expected from the command line
        if self.args.replicable:
            seed_num = 123
            random.seed(seed_num)
            np.random.seed(seed_num)
            torch.manual_seed(seed_num)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_num)
                torch.backends.cudnn.deterministic = True

    def _print_args(self):
        print("[LOG] {}".format('=' * 80))
        print("[LOG] {: >15}: '{}'".format("Experiment ID", self.experiment.id))
        print("[LOG] {: >15}: '{}'".format("Description", self.experiment.description))
        for key, val in vars(self.data.text).items():
            print("[LOG] {: >15}: {}".format(key, val))
        print("[LOG] {: >15}: '{}'".format("Modeling", self.model.name))
        #print("[LOG] {: >15}: '{}'".format("Training", vars(self.training)))
        #print("[LOG] {: >15}: '{}'".format("GPUs avaliable", self.optim.n_gpu))
        print("[LOG] {}".format('=' * 80))