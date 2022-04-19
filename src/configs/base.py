import sys
import torch
import os
sys.path.insert(0, '..')

class Object():
    def __init__(self):
        pass

class BaseConfig():
    def __init__(self):

        # Meta
        self.exp_num = 'base'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainset_split = 0.9
        self.model_save_path = '/media/administrator/1305D8BDB8D46DEE/5329/ckpt'
        self.prediction_save_path = '/media/administrator/1305D8BDB8D46DEE/5329/predicts'
        self.img_root = '/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data'
        self.label_root = '/media/administrator/1305D8BDB8D46DEE/5329/'
        
        # train path
        self.cap_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/train_caption_emb.txt'
        self.img_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_img_feature3.npy'
        self.det_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_det_feature.npy'
        # test path
        self.test_cap_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/test_caption_emb.txt'
        self.test_img_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_test_img_feature3.npy'
        self.test_det_feature_root = '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_test_det_feature.npy'

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
        self.image.augment = dict(flip_prob=0.05, rotate_prob=[0.9, 0.05, 0.05])

        # Detection model training
        self.detection = Object()
        self.detection.input_dim = 91
        self.detection.hidden_dim = 36
        self.detection.learning_rate = 1e-3
        self.detection.weight_decay = 0
        self.detection.batch_size = 32
        self.detection.total_epoch = 15
        
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
     