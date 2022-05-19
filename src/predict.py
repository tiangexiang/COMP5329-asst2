# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from combine_model import CombineModel
from caption_model import Caption
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch.nn.functional as F

import csv
from myutils import evaluate, parse_configs

header = ['ImageID', 'Labels']

def predict(config, args):

    caption_model = Caption(config.caption.input_dim, config.caption.hidden_dim, body=config.caption.body).to('cuda:0')
    caption_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth')))
    caption_model.eval()

    combine_model = CombineModel(config.combine.input_dim, config.combine.hidden_dim, dropout=config.combine.dropout).to(config.device)
    combine_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'combine_model.pth')))
    combine_model.eval()


    print('\n=========== Data Preparation ===========')

    test_dataset = MultiLabelDataset(img_root=None,
                                label_root=config.label_root,
                                cap_root=config.test_cap_root,
                                img_feature_root=config.test_img_feature_root,
                                phase='test',    
                                )

    test_loader = DataLoader(test_dataset,
                            batch_size = 100,
                            shuffle = False,
                            collate_fn = my_collate)

    print('\n=========== Predict ===========')
    preds = []
    outputs = []
    img_ids = []
    for batch_id, data in enumerate(tqdm(test_loader)):
        feat = data['img_features'].cuda()
       
        labels = data['labels']
        caps = data['caps']
        imgids = data['img_ids']
        img_ids += imgids

        with torch.no_grad():
            _, cap_features = caption_model(caps)
            predictions, output = combine_model([feat, cap_features])

        outputs += output.tolist()
        preds += predictions.tolist()

    # load threshold
    if os.path.exists('threshold.npy'):
        threshold = np.load('threshold.npy')
        print('Best threshold loaded!')
    else:
        threshold = [0.5 for _ in range(19)]

    # write csv
    with open(os.path.join(config.prediction_save_path, config.exp_num+'.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for n in tqdm(range(len(img_ids))):
            pred = []
            #logits.append(outputs[n])
            for i in range(1, 20):
                if i == 12:
                    continue
                if outputs[n][i-1]>threshold[i-1]:
                    pred.append(str(i))
            if len(pred) == 0:
                cur = deepcopy(outputs[n])
                idx = np.argsort(np.delete(cur, 11, 0))
                pred = [str(idx[-1]+1)]
            writer.writerow([img_ids[n], ' '.join(pred)])
if __name__ == '__main__':
    args, config = parse_configs()
    predict(config, args)
    
    