import torchvision
import torchvision.models as models
import torch
from dataset import MultiLabelDataset, my_collate
import torch.nn.functional as F
from myutils import parse_configs

from tqdm import tqdm
import os
import numpy as np

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

config = parse_configs()

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l').to('cuda:0')
yolo.eval()

yolo.model.model.model[-2].register_forward_hook(get_activation('backbone_feature'))

# print(type(model))
batch_size = 20
train_dataset = MultiLabelDataset(img_root=config.img_root,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                #'/media/administrator/1305D8BDB8D46DEE/5329/cap_embedding/'
                                )
loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size = batch_size,
                        shuffle = False,
                        collate_fn = my_collate)

train_classes = []
train_confidence = []
train_features = []
for batch_id, data in enumerate(tqdm(loader)):
    imgs = data['imgs']
    outputs = yolo(imgs)
    # collect detection
    for example in outputs.xyxy:
        now_class = []
        now_score = []
        for pred in example:
            pred = pred.tolist()
            now_class.append(pred[-1])
            now_score.append(pred[-2])
        train_classes.append(now_class)
        train_confidence.append(now_score)
    # collect features
    features = activation['backbone_feature']
    features = F.adaptive_avg_pool2d(features, 7)
    train_features.append(features.cpu().numpy())
    activation.clear()
    #break
train_features = np.concatenate(train_features, axis=0)
print('image feature size:', train_features.shape, 'saved to:', config.img_feature_root)
np.save(config.img_feature_root, train_features)
del train_features

test_dataset = MultiLabelDataset(img_root=config.img_root,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root,#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                phase='test'
                                #'/media/administrator/1305D8BDB8D46DEE/5329/cap_embedding/'
                                )
loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size = batch_size,
                        shuffle = False,
                        collate_fn = my_collate)

# test
test_classes = []
test_confidence = []
test_features = []
for batch_id, data in enumerate(tqdm(loader)):
    imgs = data['imgs']
    outputs = yolo(imgs)
    for example in outputs.xyxy:
        now_class = []
        now_score = []
        for pred in example:
            pred = pred.tolist()
            now_class.append(pred[-1])
            now_score.append(pred[-2])
        test_classes.append(now_class)
        test_confidence.append(now_score)
    # collect features
    features = activation['backbone_feature']
    features = F.adaptive_avg_pool2d(features, 7)
    test_features.append(features.cpu().numpy())
    activation.clear()
    #break
test_features = np.concatenate(test_features, axis=0)
print('image feature size:', test_features.shape, 'saved to:', config.test_img_feature_root)
np.save(config.test_img_feature_root, test_features)
del test_features

train_classes = [[int(_) for _ in x] for x in train_classes]
test_classes = [[int(_) for _ in x] for x in test_classes]
train_confs = []
for i in tqdm(range(len(train_dataset))):
    now_feature = [[] for _ in range(91)]#np.zeros(91)
    #now_feature = np.zeros(91)
    for j in range(len(train_classes[i])):
        if train_classes[i][j]>=91:
            continue
        # if now_feature[train_classes[i][j]]==0:
        #     now_feature[train_classes[i][j]]=train_confidence[i][j]
        now_feature[train_classes[i][j]].append(train_confidence[i][j])
    
    for k in range(91):
        if len(now_feature[k]) == 0:
            now_feature[k] = 0.
        else:
            now_feature[k] = np.max(now_feature[k])
    now_feature = np.array(now_feature)
    train_confs.append(now_feature)
    #break
test_confs = []
for i in tqdm(range(len(test_dataset))):
    now_feature = [[] for _ in range(91)]#
    #now_feature = np.zeros(91)
    for j in range(len(test_classes[i])):
        if test_classes[i][j]>=91:
            continue
        # if now_feature[test_classes[i][j]]==0:
        #    now_feature[test_classes[i][j]]=test_confidence[i][j]
        now_feature[test_classes[i][j]].append(test_confidence[i][j])
    
    for k in range(91):
        if len(now_feature[k]) == 0:
            now_feature[k] = 0.
        else:
            now_feature[k] = np.max(now_feature[k])
    now_feature = np.array(now_feature)

    test_confs.append(now_feature)
    #break

train_confs = np.array(train_confs) # N, 91
test_confs = np.array(test_confs) # N, 91
# train_features = np.concatenate((train_features, train_confs), axis=1)
# test_features = np.concatenate((test_features, test_confs), axis=1)
print('detection feature size:', train_confs.shape, test_confs.shape)
np.save(config.det_feature_root, train_confs)
np.save(config.test_det_feature_root, test_confs)
