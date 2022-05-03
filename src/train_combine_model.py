# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from combine_model import CombineModel
from image_model import ImageModel
from myutils import evaluate, parse_configs
from caption_model import Caption
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from asl import AsymmetricLossOptimized
from tqdm import tqdm

import torch.optim as optim

activation = {}
def get_activation(name, resize_size=None):
    def hook(model, input, output):
        feat = output.detach()
        if resize_size is not None:
            feat = F.adaptive_avg_pool2d(feat, resize_size)
        activation[name] = feat
    return hook

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l').to('cuda:0')
yolo.model.model.model = yolo.model.model.model[:-1]
yolo.eval()

c3_idx = [2, 4, 6, 8, 13, 17, 20, 23]
c3_idx_smaller_feature_map = [6, 8, 13, 20, 23]
c3_idx_smaller_feature_map_feature_1024 = [8, 23]
for idx, c in enumerate(yolo.model.model.model.children()):
    if idx in c3_idx_smaller_feature_map:
        yolo.model.model.model[idx].register_forward_hook(get_activation(str(c3_idx_smaller_feature_map.index(idx))+'_feature', 7))


# Function used to train the combine model
def train_combine_model(Combine_model, optimizer, criterion, config):


    caption_model = Caption(config.caption.input_dim, config.caption.hidden_dim).to('cuda:0')
    caption_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth')))
    caption_model.eval()

    # multi level image model
    image_models = []
    for feat_level in config.image.levels:
        image_models.append(ImageModel(config.image.level2dim[feat_level], config.image.hidden_dim, number_layers=1, dropout=config.image.dropout, head='ml', bidirectional=True).to(config.device))
        image_models[-1].load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'image_model_'+str(feat_level)+'.pth')))
        image_models[-1].eval()

    # det_model = ImageModel(config.detection.input_dim, config.detection.hidden_dim, number_layers=1, dropout=0, head='lstm', bidirectional=True).to('cuda:0')
    # det_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'detection_model.pth')), strict=False)
    # det_model.eval()


    print('\n=========== Data Preparation ===========')
    train_dataset = MultiLabelDataset(img_root=config.img_root,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root,#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                cap_root=config.cap_root,#'/media/administrator/1305D8BDB8D46DEE/5329/cap_embedding/',
                                img_feature_root=None,#config.img_feature_root,#['/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_img_feature3.npy', '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_det_feature.npy'],
                                )

    train_size = int(len(train_dataset) * config.trainset_split)
    val_size = len(train_dataset) - train_size
    train_loader = DataLoader(Subset(train_dataset, list(range(0, train_size-val_size))+list(range(train_size, len(train_dataset)))),
                        batch_size = config.combine.batch_size,
                        shuffle = True,
                        collate_fn = my_collate)
    val_loader = DataLoader(Subset(train_dataset, range(train_size-val_size, train_size)),
                            batch_size = config.combine.batch_size,
                            shuffle=False,
                            collate_fn = my_collate)
    train_loss_log = []
    train_micro_f1_log = []
    train_macro_f1_log = []
    train_weighted_f1_log = []
    train_samples_f1_log = []
    val_loss_log = []
    val_micro_f1_log = []
    val_macro_f1_log = []
    val_weighted_f1_log = []
    val_samples_f1_log = []
    print('Training Set Size:', train_size)
    print("Validation Set Size:", val_size)
    print('\n=========== Training ===========')
    best_val_loss = 100
    for epoch in range(config.combine.total_epoch):
        Combine_model.train()
        epoch_loss = 0
        # Train
        train_preds = []
        train_labels = []
        for batch_id, data in enumerate(tqdm(train_loader)):
            labels = data['labels']
            imgs = data['imgs']
            caps = data['caps']
            optimizer.zero_grad()
            #print(imgs.shape)
            features = []
            with torch.no_grad():
                _ = yolo(imgs)
                for i in range(len(config.image.levels)):
                    features.append(image_models[i](activation[str(config.image.levels[i])+'_feature'].to('cuda:0'))[1])
                # _, img_features_4 = image_model_4(activation['4_feature'].to('cuda:0'))
                # _, img_features_1 = image_model_1(activation['1_feature'].to('cuda:0'))
                _, cap_features = caption_model(caps)

            predictions, output = Combine_model(features + [cap_features])
            activation.clear()
            #predictions, output = Combine_model(data['img_features'].to(device), data['cap_features'].to(device))
            labels = labels.to(config.device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*data['labels'].shape[0]
            train_preds += predictions.tolist()
            train_labels += labels.tolist()
        # Val
        Combine_model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_scores = []
        for batch_id, data in enumerate(tqdm(val_loader)):
            labels = data['labels']
            imgs = data['imgs']
            caps = data['caps']
            features = []
            with torch.no_grad():
                _ = yolo(imgs)
                for i in range(len(config.image.levels)):
                    features.append(image_models[i](activation[str(config.image.levels[i])+'_feature'].to('cuda:0'))[1])
                # _, img_features_4 = image_model_4(activation['4_feature'].to('cuda:0'))
                # _, img_features_1 = image_model_1(activation['1_feature'].to('cuda:0'))
                _, cap_features = caption_model(caps)

            predictions, output = Combine_model(features + [cap_features])
            activation.clear()
            labels = labels.to(config.device)
            loss = criterion(output, labels)
            val_loss += loss.item()*data['labels'].shape[0]
            val_preds += predictions.tolist()
            val_labels += labels.tolist()
            val_scores += output.tolist()
        train_micro_f1, train_macro_f1, train_weighted_f1, train_samples_f1 = evaluate(train_preds, train_labels)
        val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds, val_labels)
        epoch_loss /= train_size
        val_loss /= val_size
        print("Epoch: %d, train loss: %.4f, val loss: %.4f" % (epoch+1, epoch_loss, val_loss))
        print("Train: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (train_micro_f1, train_macro_f1, train_weighted_f1, train_samples_f1))
        print("Val: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1))
        train_loss_log.append(epoch_loss)
        train_micro_f1_log.append(train_micro_f1)
        train_macro_f1_log.append(train_macro_f1)
        train_weighted_f1_log.append(train_weighted_f1)
        train_samples_f1_log.append(train_samples_f1)
        val_loss_log.append(val_loss)
        val_micro_f1_log.append(val_micro_f1)
        val_macro_f1_log.append(val_macro_f1)
        val_weighted_f1_log.append(val_weighted_f1)
        val_samples_f1_log.append(val_samples_f1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(Combine_model.state_dict(), os.path.join(config.model_save_path, config.exp_num, 'combine_model.pth'))
            #torch.save(Combine_model, 'best_model.pt')
    # Val
    Combine_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'combine_model.pth')))
    #Combine_model = torch.load('best_model.pt')
    print('\n=========== Validation ===========')
    Combine_model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_scores = []
    for batch_id, data in enumerate(tqdm(val_loader)):
        labels = data['labels']
        imgs = data['imgs']
        caps = data['caps']
        features = []
        with torch.no_grad():
            _ = yolo(imgs)
            for i in range(len(config.image.levels)):
                features.append(image_models[i](activation[str(config.image.levels[i])+'_feature'].to('cuda:0'))[1])
                # _, img_features_4 = image_model_4(activation['4_feature'].to('cuda:0'))
                # _, img_features_1 = image_model_1(activation['1_feature'].to('cuda:0'))
            _, cap_features = caption_model(caps)

        predictions, output = Combine_model(features + [cap_features])
        activation.clear()
        labels = labels.to(config.device)
        loss = criterion(output, labels)
        val_loss += loss.item()*data['labels'].shape[0]
        val_preds += predictions.tolist()
        val_labels += labels.tolist()
        val_scores += output.tolist()
    val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds, val_labels)
    val_loss /= val_size
    print("val loss: %.4f" % (val_loss))
    print("Val: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1))
    return train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log

if __name__ == '__main__':
    args, config = parse_configs()

    Combine_model = CombineModel(config.combine.input_dim, dropout=config.combine.dropout).to(config.device)
    criterion = nn.BCELoss()
    #criterion = AsymmetricLossOptimized()

    optimizer = optim.Adam(Combine_model.parameters(), lr=config.combine.learning_rate, weight_decay=config.combine.weight_decay)
    train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log = train_combine_model(Combine_model, optimizer, criterion, config)
