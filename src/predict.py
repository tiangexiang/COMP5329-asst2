# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from combine_model import CombineModel
from image_model import ImageModel
from caption_model import Caption
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from asl import AsymmetricLossOptimized
from tqdm import tqdm
import csv
from myutils import evaluate, parse_configs

header = ['ImageID', 'Labels']

def predict(config):

    caption_model = Caption(config.caption.input_dim, config.caption.hidden_dim).to('cuda:0')
    caption_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth')))
    caption_model.eval()

    image_model = ImageModel(config.image.input_dim, config.image.hidden_dim, number_layers=1, dropout_rate=0, head='ml', bidirectional=True).to('cuda:0')
    image_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'image_model.pth')))
    image_model.eval()

    det_model = ImageModel(config.detection.input_dim, config.detection.hidden_dim, number_layers=1, dropout_rate=0, head='lstm', bidirectional=True).to('cuda:0')
    det_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'detection_model.pth')))
    det_model.eval()

    combine_model = CombineModel(config.combine.input_dim, dropout=config.combine.dropout).to('cuda:0')
    combine_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'combine_model.pth')))
    combine_model.eval()


    print('\n=========== Data Preparation ===========')
    test_dataset = MultiLabelDataset(img_root=None,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root,#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                cap_root=config.test_cap_root,#'/media/administrator/1305D8BDB8D46DEE/5329/cap_embedding/',
                                img_feature_root=config.test_img_feature_root,#['/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_test_img_feature3.npy', '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_test_det_feature.npy'],
                                det_feature_root=config.test_det_feature_root,#None,#'/media/administrator/1305D8BDB8D46DEE/5329/features/train_cap_features.npy'
                                phase='test'
                                )

    test_loader = DataLoader(test_dataset,
                            batch_size = 100,
                            shuffle = False,
                            collate_fn = my_collate)

    print('\n=========== Predict ===========')
    preds = []
    img_ids = []
    for batch_id, data in enumerate(tqdm(test_loader)):
        labels = data['labels']
        imgs = data['img_features']
        caps = data['caps']
        imgids = data['img_ids']
        img_ids += imgids

        with torch.no_grad():
            _, img_features = image_model(imgs[0].to('cuda:0'))
            _, det_features = det_model(imgs[1].to('cuda:0'))
            _, cap_features = caption_model(caps)
            predictions, output = combine_model(img_features, det_features, cap_features)
        
        preds += predictions.tolist()

    # write csv
    with open(os.path.join(config.prediction_save_path, config.exp_num+'.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for n in tqdm(range(len(img_ids))):
            pred = []
            for i in range(1, 20):
                if preds[n][i-1]>0.5:
                    pred.append(str(i))
            writer.writerow([img_ids[n], ' '.join(pred)])


if __name__ == '__main__':
    config = parse_configs()
    predict(config)
    
    