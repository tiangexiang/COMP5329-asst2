# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from combine_model import CombineModel
from image_model import ImageModel
from linear_model import LinearModel
from myutils import evaluate, parse_configs
from caption_model import Caption
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from asl import AsymmetricLossOptimized
from tqdm import tqdm
import numpy as np
import torch.optim as optim

def find_threshold(config, args):

    caption_model = Caption(config.caption.input_dim, config.caption.hidden_dim).to('cuda:0')
    caption_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth')))
    caption_model.eval()

    Combine_model = CombineModel(config.combine.input_dim, dropout=config.combine.dropout).to(config.device)
    Combine_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'combine_model.pth')))
    Combine_model.eval()

    criterion = nn.BCELoss()

    print('\n=========== Data Preparation ===========')
    train_dataset = MultiLabelDataset(img_root=None,
                                label_root=config.label_root,
                                cap_root=config.cap_root,
                                img_feature_root=config.img_feature_root,
                                )

    train_size = int(len(train_dataset) * config.trainset_split)
    val_size = len(train_dataset) - train_size

    val_loader = DataLoader(train_dataset,
                            #Subset(train_dataset, range(train_size-val_size, train_size)),
                            batch_size = config.combine.batch_size,
                            shuffle=False,
                            collate_fn = my_collate)

    print('\n=========== Validation ===========')
    val_loss = 0
    val_preds = []
    val_labels = []
    val_scores = []
    logits = []
    for batch_id, data in enumerate(tqdm(val_loader)):
        labels = data['labels']
        caps = data['caps']
        feat = data['img_features'].cuda()

        with torch.no_grad():
            _, cap_features = caption_model(caps)
            predictions, output = Combine_model([feat, cap_features])
        
        logits.append(output)
        labels = labels.to(config.device)
        loss = criterion(output, labels)
        val_loss += loss.item()*data['labels'].shape[0]
        val_preds += predictions.tolist()
        val_labels += labels.tolist()
        val_scores += output.tolist()
    val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds, val_labels)
    val_loss /= val_size
    print("val loss: %.4f" % (val_loss))
    print("original samples f1: %.4f" % (val_samples_f1))
    val_labels = np.array(val_labels)
    val_scores = np.array(val_scores)
    
    best_t = [None for _ in range(19)]
    for class_idx in range(19):
        best_f1 = -1
        for t in tqdm(np.arange(0.0, 1.0, 0.005)):
            val_preds = val_scores >= t
            micro_f1,macro_f1,weighted_f1, val_samples_f1 = evaluate(val_preds[:,class_idx], val_labels[:,class_idx])
            if weighted_f1 > best_f1:
                best_f1 = weighted_f1
                best_t[class_idx] = t
        print("[class %d] best f1: %.4f, t: %.4f" % (class_idx+1, best_f1, best_t[class_idx]))
    best_t = np.array(best_t)
    print(best_t.shape)
    np.save('threshold.npy', best_t)

if __name__ == '__main__':
    args, config = parse_configs()
    
    find_threshold(config, args)
