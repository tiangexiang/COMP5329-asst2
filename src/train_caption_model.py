# Function used to train the caption model
from dataset import MultiLabelDataset, my_collate
import torch.nn as nn
from caption_model import Caption
from myutils import evaluate, parse_configs
import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim


def train_caption_model(Caption_model, optimizer, criterion, config):
    print('\n=========== Data Preparation ===========')
    train_dataset = MultiLabelDataset(img_root=None,
                                      label_root=config.label_root,
                                      cap_root=config.cap_root,
                                     )
    train_size = int(len(train_dataset) * config.trainset_split)
    val_size = len(train_dataset) - train_size
    train_loader = torch.utils.data.DataLoader(Subset(train_dataset, range(0, train_size)),
                        batch_size = config.caption.batch_size,
                        shuffle = True,
                        collate_fn = my_collate)
    val_loader = torch.utils.data.DataLoader(Subset(train_dataset, range(train_size, len(train_dataset))),
                            batch_size = config.caption.batch_size,
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
    for epoch in range(config.caption.total_epoch):
        Caption_model.train()
        epoch_loss = 0
        # Train
        train_preds = []
        train_labels = []
        for batch_id, data in enumerate(tqdm(train_loader)):
            captions, labels = data['caps'], data['labels']
            optimizer.zero_grad()
            predictions, output = Caption_model(captions)
            labels = labels.to(config.device)
            loss = criterion(output.float(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(captions)
            train_preds += predictions.tolist()
            train_labels += labels.tolist()

        # Val
        Caption_model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        for batch_id, data in enumerate(tqdm(val_loader)):
            captions, labels = data['caps'], data['labels']
            predictions, output = Caption_model(captions)
            labels = labels.to(config.device)
            loss = criterion(output, labels)
            val_loss += loss.item()*len(captions)
            val_preds += predictions.tolist()
            val_labels += labels.tolist()
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
            Caption_model.half()
            torch.save(Caption_model.state_dict(), os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth'))
            Caption_model.float()

    # Val
    Caption_model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.exp_num, 'caption_model.pth')))
    print('\n=========== Validation ===========')
    Caption_model.float().eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    val_scores = []
    for batch_id, data in enumerate(tqdm(val_loader)):
        captions, labels = data['caps'], data['labels']
        predictions, output = Caption_model(captions)
        labels = labels.to(config.device)
        loss = criterion(output, labels)
        val_loss += loss.item()*len(captions)
        val_preds += predictions.tolist()
        val_labels += labels.tolist()
        val_scores += output.tolist()
    val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1 = evaluate(val_preds, val_labels)
    val_loss /= val_size
    print("val loss: %.4f" % (val_loss))
    print("Val: micro f1: %.4f, macro f1: %.4f, weighted f1: %.4f, samples f1: %.4f" % (val_micro_f1, val_macro_f1, val_weighted_f1, val_samples_f1))
    return train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log


if __name__ == '__main__':
    
    # Train a caption model
    args, config = parse_configs()
    
    Caption_model = Caption(config.caption.input_dim, config.caption.hidden_dim, body=config.caption.body, sigmoid=True).to(config.device)
    Caption_model = Caption_model#.half()
    criterion = nn.BCELoss()

    optimizer = optim.Adam(Caption_model.parameters(), lr=config.caption.learning_rate, weight_decay=config.caption.weight_decay)
    train_loss_log, train_micro_f1_log, train_macro_f1_log, train_weighted_f1_log, train_samples_f1_log, val_loss_log, val_micro_f1_log, val_macro_f1_log, val_weighted_f1_log, val_samples_f1_log = train_caption_model(Caption_model, optimizer, criterion, config)