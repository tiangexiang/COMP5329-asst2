# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from image_model import ImageModel
from myutils import evaluate, parse_configs
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from asl import AsymmetricLossOptimized
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
    if idx in c3_idx_smaller_feature_map_feature_1024:
        yolo.model.model.model[idx].register_forward_hook(get_activation(str(c3_idx_smaller_feature_map.index(idx))+'_feature', 7))


# Function used to train the image model
def train_image_model(Image_models, optimizers, criterion, config, args):
    print('\n=========== Data Preparation ===========')

    train_dataset = MultiLabelDataset(img_root=config.img_root,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root,#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                cap_root=None,
                                img_feature_root=None,#config.img_feature_root,#'/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_img_feature3.npy',
                                img_feature_level=None,#args.level
                                #cap_feature_root='/media/administrator/1305D8BDB8D46DEE/5329/features/train_cap_features.npy'
                                )
    
    train_size = int(len(train_dataset) * config.trainset_split)
    val_size = len(train_dataset) - train_size
    train_loader = DataLoader(Subset(train_dataset, range(0, train_size)),
                        batch_size = config.image.batch_size,
                        shuffle = True,
                        collate_fn = my_collate)
    val_loader = DataLoader(Subset(train_dataset, range(train_size, len(train_dataset))),
                            batch_size = config.image.batch_size,
                            shuffle=False,
                            collate_fn = my_collate)
    train_loss_log = [[] for _ in range(len(config.image.levels))]
    train_micro_f1_log = [[] for _ in range(len(config.image.levels))]
    train_macro_f1_log = [[] for _ in range(len(config.image.levels))]
    train_weighted_f1_log = [[] for _ in range(len(config.image.levels))]
    train_samples_f1_log = [[] for _ in range(len(config.image.levels))]
    val_loss_log = [[] for _ in range(len(config.image.levels))]
    val_micro_f1_log = [[] for _ in range(len(config.image.levels))]
    val_macro_f1_log = [[] for _ in range(len(config.image.levels))]
    val_weighted_f1_log = [[] for _ in range(len(config.image.levels))]
    val_samples_f1_log = [[] for _ in range(len(config.image.levels))]
    print('Training Set Size:', train_size)
    print("Validation Set Size:", val_size)
    print('\n=========== Training ===========')
    best_val_loss = [100 for _ in range(len(config.image.levels))]
    for epoch in range(config.image.total_epoch):
        for i in range(len(config.image.levels)):
            Image_models[i].train()
        epoch_loss = [0 for _ in range(len(config.image.levels))]
        # Train
        train_preds = [[] for _ in range(len(config.image.levels))]
        train_labels = []
        count = 1
        pbar = tqdm(train_loader)
        for batch_id, data in enumerate(pbar):
            #pbar.set_description("[TRAIN] Epoch %d Loss: %.4f" % (epoch, epoch_loss / count))
            count += 1
            labels = data['labels']
            imgs = data['imgs'].to(config.device) if data['imgs'] is not None else data['img_features']#[:,1024:]

            with torch.no_grad():
                _ = yolo(imgs)

            for i in range(len(config.image.levels)):
                optimizers[i].zero_grad()
            
                predictions, output = Image_models[i](activation[str(config.image.levels[i])+'_feature'], augment=config.image.augment)
                labels = labels.to(config.device)
                loss = criterion(output, labels)
                loss.backward()
                optimizers[i].step()
                epoch_loss[i] += loss.item()*labels.shape[0]
                train_preds[i] += predictions.tolist()
            activation.clear()
            train_labels += labels.tolist()

        # Val
        for i in range(len(config.image.levels)):
            Image_models[i].eval()
        val_loss = [0 for _ in range(len(config.image.levels))]
        val_preds = [[] for _ in range(len(config.image.levels))]
        val_scores = [[] for _ in range(len(config.image.levels))]
        val_labels = []
        
        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(val_loader)):
                labels = data['labels']
                imgs = data['imgs'].to(config.device) if data['imgs'] is not None else data['img_features']#[:,1024:]
                _ = yolo(imgs)
                for i in range(len(config.image.levels)):
                    predictions, output = Image_models[i](activation[str(config.image.levels[i])+'_feature'], augment=None)
                    labels = labels.to(config.device)
                    loss = criterion(output, labels)
                    val_loss[i] += loss.item()*imgs.shape[0]
                    val_preds[i] += predictions.tolist()
                    val_scores[i] += output.tolist()
                activation.clear()
                val_labels += labels.tolist()
        
        for i in range(len(config.image.levels)):
            train_micro_f1, train_macro_f1, train_weighted_f1, train_samples_f1 = evaluate(train_preds[i], train_labels)
            val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds[i], val_labels)
            epoch_loss[i] /= train_size
            val_loss[i] /= val_size
            print('=== model at level '+str(config.image.levels[i])+' ===')
            print("Epoch: %d, train loss: %.4f, val loss: %.4f" % (epoch+1, epoch_loss[i], val_loss[i]))
            print("Train: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (train_micro_f1, train_macro_f1, train_weighted_f1, train_samples_f1))
            print("Val: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1))
            print('=== ============= ===')
            train_loss_log[i].append(epoch_loss)
            train_micro_f1_log[i].append(train_micro_f1)
            train_macro_f1_log[i].append(train_macro_f1)
            train_weighted_f1_log[i].append(train_weighted_f1)
            train_samples_f1_log[i].append(train_samples_f1)
            val_loss_log[i].append(val_loss)
            val_micro_f1_log[i].append(val_micro_f1)
            val_macro_f1_log[i].append(val_macro_f1)
            val_weighted_f1_log[i].append(val_weighted_f1)
            val_samples_f1_log[i].append(val_samples_f1)

            if val_loss[i] < best_val_loss[i]:
                best_val_loss[i] = val_loss[i]
                #torch.save(Image_model, 'best_model.pt')
                torch.save(Image_models[i].state_dict(), os.path.join(config.model_save_path, config.exp_num, 'image_model_'+str(config.image.levels[i])+'.pth'))
    # Val
    #Image_model = torch.load('best_model.pt')
    for i in range(len(config.image.levels)):
        Image_models[i].load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num,'image_model_'+str(config.image.levels[i])+'.pth')))
        Image_models[i].eval()
    print('\n=========== Validation ===========')
    val_loss = [0 for _ in range(len(config.image.levels))]
    val_preds = [[] for _ in range(len(config.image.levels))]
    val_labels = []
    val_scores = [[] for _ in range(len(config.image.levels))]
    with torch.no_grad():
        for batch_id, data in enumerate(tqdm(val_loader)):
            labels = data['labels']
            imgs = data['imgs'].to(config.device) if data['imgs'] is not None else data['img_features']#[:,1024:]
            _ = yolo(imgs)

            for i in range(len(config.image.levels)):
                predictions, output = Image_models[i](activation[str(config.image.levels[i])+'_feature'], augment=None)
                labels = labels.to(config.device)
                loss = criterion(output, labels)
                val_loss[i] += loss.item()*imgs.shape[0]
                val_preds[i] += predictions.tolist()
                val_scores[i] += output.tolist()
            activation.clear()
            val_labels += labels.tolist()

    for i in range(len(config.image.levels)):
        val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds[i], val_labels)
        val_loss[i] /= val_size
        print('=== model at level '+str(config.image.levels[i])+' ===')
        print("val loss: %.4f" % (val_loss[i]))
        print("Val: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1))
        print('=== ============= ===')
    return train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log

if __name__ == '__main__':
    # Train an image model
    args, config = parse_configs()

    criterion = nn.BCELoss()
    #criterion = AsymmetricLossOptimized()
    
    Image_models, optimizers = [], []
    for feat_level in config.image.levels:
        Image_models.append(ImageModel(config.image.level2dim[feat_level], config.image.hidden_dim, number_layers=1, dropout=config.image.dropout, head='ml', bidirectional=True).to(config.device))
        optimizers.append(optim.Adam(Image_models[-1].parameters(), lr=config.image.learning_rate, weight_decay=config.image.weight_decay))
    
    train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log = train_image_model(Image_models, optimizers, criterion, config, args)
