# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from image_model import ImageModel
from myutils import evaluate, parse_configs
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim


# Function used to train the image model
def train_image_model(Image_model, optimizer, criterion, config):
    print('\n=========== Data Preparation ===========')
    train_dataset = MultiLabelDataset(img_root=None,#'/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                label_root=config.label_root,#'/media/administrator/1305D8BDB8D46DEE/5329/',
                                det_feature_root=config.det_feature_root#'/media/administrator/1305D8BDB8D46DEE/5329/features/yolov5_train_det_feature.npy',
                                #det_feature_root=None
                                )
    train_size = int(len(train_dataset) * config.trainset_split)
    val_size = len(train_dataset) - train_size
    train_loader = DataLoader(Subset(train_dataset, range(0, train_size)),
                        batch_size = config.detection.batch_size,
                        shuffle = True,
                        collate_fn = my_collate)
    val_loader = DataLoader(Subset(train_dataset, range(train_size, len(train_dataset))),
                            batch_size = config.detection.batch_size,
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
    for epoch in range(config.detection.total_epoch):
        Image_model.train()
        epoch_loss = 0
        # Train
        train_preds = []
        train_labels = []
        count = 1
        pbar = tqdm(train_loader)
        for batch_id, data in enumerate(pbar):
            pbar.set_description("[TRAIN] Epoch %d Loss: %.4f" % (epoch, epoch_loss / count))
            count += 1
            optimizer.zero_grad()
            labels = data['labels']
            imgs = data['imgs'] if data['imgs'] is not None else data['det_features']#[:,1024:]
            predictions, output = Image_model(imgs.to(config.device))
            labels = labels.to(config.device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*imgs.shape[0]
            train_preds += predictions.tolist()
            train_labels += labels.tolist()

        # Val
        Image_model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_scores = []
        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(val_loader)):
                labels = data['labels']
                imgs = data['imgs'] if data['imgs'] is not None else data['det_features']#[:,1024:]
                predictions, output = Image_model(imgs.to(config.device))
                labels = labels.to(config.device)
                loss = criterion(output, labels)
                val_loss += loss.item()*imgs.shape[0]
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
            torch.save(Image_model.state_dict(), os.path.join(config.model_save_path, config.exp_num, 'detection_model.pth'))
            #torch.save(Image_model, 'best_model.pt')
    # Val
    Image_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'detection_model.pth')))
    #Image_model = torch.load('best_model.pt')
    print('\n=========== Validation ===========')
    Image_model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_scores = []
    with torch.no_grad():
        for batch_id, data in enumerate(tqdm(val_loader)):
            labels = data['labels']
            imgs = data['imgs'] if data['imgs'] is not None else data['det_features']#[:,1024:]
            predictions, output = Image_model(imgs.to(config.device))
            labels = labels.to(config.device)
            loss = criterion(output, labels)
            val_loss += loss.item() * imgs.shape[0]
            val_preds += predictions.tolist()
            val_labels += labels.tolist()
            val_scores += output.tolist()
    val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds, val_labels)
    val_loss /= val_size
    print("val loss: %.4f" % (val_loss))
    print("Val: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1))
    return train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log

if __name__ == '__main__':
    # Train a detection model
    config = parse_configs()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dim = 91#1792
    # learning_rate = 1e-3
    # weight_decay = 0
    # batch_size = 32
    # total_epoch = 15
 
    Image_model = ImageModel(config.detection.input_dim, config.detection.hidden_dim, number_layers=1, dropout=0, head='lstm', bidirectional=True).to(config.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(Image_model.parameters(), lr=config.detection.learning_rate, weight_decay=config.detection.weight_decay)
    train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log = train_image_model(Image_model, optimizer, criterion, config)


    # save model
    # if not os.path.exists(os.path.join(model_save_path, exp_num)):
    #     os.mkdir(os.path.join(model_save_path, exp_num))

    # torch.save(Image_model.state_dict(), os.path.join(model_save_path, exp_num, 'detection_model.pth'))
    # print('detection model saved in '+os.path.join(model_save_path, exp_num, 'detection_model.pth'))

    
    