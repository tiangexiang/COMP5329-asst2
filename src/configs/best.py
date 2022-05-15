import sys
import torch
import os
from .base import BaseConfig
sys.path.insert(0, '..')

class Object():
    def __init__(self):
        pass

class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        # Meta
        self.exp_num = 'best'
        self.trainset_split = 0.95
        # train path
        self.cap_root = '/media/sda5/USYD/5329/features/train_caption_emb.txt'
        self.img_feature_root = '/media/sda5/USYD/5329/features/train_teacher_logits_7.npy'
        # test path
        self.test_cap_root = '/media/sda5/USYD/5329/features/test_caption_emb.txt'
        self.test_img_feature_root = '/media/sda5/USYD/5329/features/test_teacher_logits_7.npy'


        if not os.path.exists(os.path.join(self.model_save_path, self.exp_num)):
            os.mkdir(os.path.join(self.model_save_path, self.exp_num))

        # Caption model training
        self.caption = Object()
        self.caption.input_dim = 150
        self.caption.hidden_dim = 128#128
        self.caption.learning_rate = 1e-3
        self.caption.weight_decay = 0
        self.caption.batch_size = 256
        self.caption.total_epoch = 15
        self.caption.body = 'lstm'

        # Combine model training
        self.combine = Object()
        self.combine.input_dim = 80*2+19
        self.combine.learning_rate = 1e-3
        self.combine.weight_decay = 0
        self.combine.batch_size = 32
        self.combine.total_epoch = 50
        self.combine.dropout=0.2
        self.combine.flooding=0.035 # https://arxiv.org/abs/2002.08709

if __name__ == '__main__':
    a = BaseConfig()
    print(a.caption_model_config.embedding_dim)
     