import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from ml_decoder import MLDecoder
import numpy as np

class ImageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_layers=1, bidirectional=True, dropout=0, num_class=19, head='lstm'):
        super(ImageModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_class = num_class
        self.bidirectional = bidirectional
        self.head = head
        #self.backbone = models.efficientnet_b4(pretrained=True)

        if head == 'lstm':

            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=number_layers, bidirectional=bidirectional, dropout=0)
            if self.bidirectional:
                self.output = nn.Sequential(nn.Linear(hidden_dim*2*num_class, self.num_class),
                                        nn.Sigmoid())
            else:
                self.output = nn.Sequential(nn.BatchNorm1d(input_dim),
                                        nn.Linear(input_dim, self.num_class),
                                        nn.Sigmoid())
                # self.output = nn.Sequential(nn.Linear(hidden_dim*num_class, self.num_class),
                #                         nn.Sigmoid())
        else:

            self.ml_decoder_head = nn.Sequential(#nn.BatchNorm2d(input_dim),
                                #nn.Dropout(0.1),
                                MLDecoder(num_class, initial_num_features=input_dim, dim_feedforward=512, decoder_embedding=768//2)) # initilization
            # self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=number_layers, bidirectional=bidirectional, dropout=dropout_rate)
            # self.output = nn.Sequential(nn.Linear(hidden_dim*2*num_class, self.num_class),
            #                             nn.Sigmoid())


    def forward(self, x, augment=None):
        # image_features = self.backbone.features(x)
        # image_features = F.adaptive_avg_pool2d(image_features, 1)

        if self.head == 'lstm':
            if len(x.shape) == 4:
                x = F.adaptive_avg_pool2d(x, 1).squeeze()
            # image_features [batch_size, feature_size]
            image_features = x.view(x.shape[0], 1, -1)
            # image_features [batch_size, seq_len(num_class), feature_size]
            image_features = image_features.expand(image_features.shape[0], self.num_class, image_features.shape[-1])
            #print(image_features.shape)
            # output [batch_size, seq_len, num_directions * hidden_size]
            output, (_, _) = self.lstm(image_features)
            # output [batch_size, seq_len * num_directions * hidden_size]

            output = output.reshape(output.shape[0], -1)
            output = self.output(output)
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(-1).unsqueeze(-1)
            if augment is not None and not self.training:
                if np.random.uniform() < augment['flip_prob']:
                    x = torch.fliplr(x)
                if np.random.uniform() < augment['flip_prob']:
                    x = torch.flipud(x)
                rotidx = np.random.choice([0, 1, 2], 1, False, p=augment['rotate_prob'])
                #print(rotidx, rotidx.shape)
                x = torch.rot90(x, rotidx[0], (-2, -1))
            x = F.dropout2d(x, self.dropout)
            output = self.ml_decoder_head(x)

            output = torch.sigmoid(output)
        
        # predictions = output > 0.5
        predictions = output > 0.5
        return predictions, output


from sklearn.metrics import f1_score
def evaluate(predictions, labels):
    if len(predictions)==0:
        return 0
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    samples_f1 = f1_score(labels, predictions, average='samples')
    return micro_f1, macro_f1, weighted_f1, samples_f1