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
        self.exp_num = '3'

        # train path
        self.cap_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/train_caption_emn999.txt'
        #self.img_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_img_feature999.npy'
        #self.det_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_det_feature999.npy'
        # test path
        self.test_cap_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/test_caption_emb999.txt'
        #self.test_img_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_test_img_feature999.npy'
        #self.test_det_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_test_det_feature999.npy'


        if not os.path.exists(os.path.join(self.model_save_path, self.exp_num)):
            os.mkdir(os.path.join(self.model_save_path, self.exp_num))

        # Caption model training
        self.caption = Object()
        self.caption.embedding_dim = 150
        self.caption.hidden_dim = 128
        self.caption.learning_rate = 1e-3
        self.caption.weight_decay = 0
        self.caption.batch_size = 256
        self.caption.total_epoch = 10

        # Image model training
        self.image = Object()
        self.image.input_dim = 1024
        self.image.hidden_dim = 64
        self.image.learning_rate =1e-4
        self.image.weight_decay = 0
        self.image.batch_size = 32
        self.image.total_epoch = 30
        self.image.dropout = 0.2
        self.image.augment = dict(flip_prob=0.1, rotate_prob=[0.9, 0.05, 0.05])

        # Detection model training
        self.detection = Object()
        self.detection.input_dim = 91
        self.detection.hidden_dim = 36
        self.detection.learning_rate = 1e-3
        self.detection.weight_decay = 0
        self.detection.batch_size = 32
        self.detection.total_epoch = 30
        
        # Combine model training
        self.combine = Object()
        self.combine.input_dim = 19*3 #2048
        self.combine.learning_rate =1e-3
        self.combine.weight_decay = 0
        self.combine.batch_size = 32
        self.combine.total_epoch = 50
        self.combine.dropout=0.1

if __name__ == '__main__':
    a = BaseConfig()
    print(a.caption_model_config.embedding_dim)
     