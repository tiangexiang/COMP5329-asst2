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


# Function used to train the combine model
def train_combine_model(Combine_model, optimizer, criterion, config):


    caption_model = Caption(config.caption.input_dim, config.caption.hidden_dim).to('cuda:0')
    caption_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth')))
    caption_model.eval()

    image_model = ImageModel(config.image.input_dim, config.image.hidden_dim, number_layers=1, dropout_rate=0, head='ml', bidirectional=True).to('cuda:0')
    image_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'image_model.pth')))
    image_model.eval()

    det_model = ImageModel(config.detection.input_dim, config.detection.hidden_dim, number_layers=1, dropout_rate=0, head='lstm', bidirectional=True).to('cuda:0')
    det_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'detection_model.pth')))
    det_model.eval()


    print('\n=========== Data Preparation ===========')
    train_dataset = MultiLabelDataset(img_root=None,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root,#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                cap_root=config.cap_root,#'/media/administrator/1305D8BDB8D46DEE/5329/cap_embedding/',
                                img_feature_root=config.img_feature_root,#['/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_img_feature3.npy', '/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_det_feature.npy'],
                                det_feature_root=config.det_feature_root
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
            imgs = data['img_features']
            dets = data['det_features']
            caps = data['caps']
            optimizer.zero_grad()

            with torch.no_grad():
               _, img_features = image_model(imgs.to('cuda:0'))
               _, det_features = det_model(dets.to('cuda:0'))
               _, cap_features = caption_model(caps)

            predictions, output = Combine_model(img_features, det_features, cap_features)
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
            imgs = data['img_features']
            dets = data['det_features']
            caps = data['caps']

            with torch.no_grad():
               _, img_features = image_model(imgs.to('cuda:0'))
               _, det_features = det_model(dets.to('cuda:0'))
               _, cap_features = caption_model(caps)

            #print(det_features.shape, cap_features.shape)
            predictions, output = Combine_model(img_features, det_features, cap_features)
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
        imgs = data['img_features']
        dets = data['det_features']
        caps = data['caps']

        with torch.no_grad():
            _, img_features = image_model(imgs.to('cuda:0'))
            _, det_features = det_model(dets.to('cuda:0'))
            _, cap_features = caption_model(caps)

        predictions, output = Combine_model(img_features, det_features, cap_features)
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

    # Train an image model
    # import torch.optim as optim
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dim = 19*3 #2048
    # learning_rate =1e-3
    # weight_decay = 0
    # batch_size = 32
    # total_epoch = 100
    Combine_model = CombineModel(config.combine.input_dim, dropout=config.combine.dropout).to(config.device)
    criterion = nn.BCELoss()
    #criterion = AsymmetricLossOptimized()

    optimizer = optim.Adam(Combine_model.parameters(), lr=config.combine.learning_rate, weight_decay=config.combine.weight_decay)
    train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log = train_combine_model(Combine_model, optimizer, criterion, config)

    # # save model
    # if not os.path.exists(os.path.join(model_save_path, exp_num)):
    #     os.mkdir(os.path.join(model_save_path, exp_num))

    # torch.save(Combine_model.state_dict(), os.path.join(model_save_path, exp_num, 'combine_model.pth'))
    # print('Combine model saved in '+os.path.join(model_save_path, exp_num, 'combine_model.pth'))

    
    