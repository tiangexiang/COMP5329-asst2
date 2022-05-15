import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from ml_decoder import MLDecoder

class CombineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, number_layers=1, dropout_rate=0, num_class=19, head='lstm'):
        super(CombineModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.head = head
        #self.bidirectional = bidirectional

        self.fc = nn.Sequential(nn.Dropout(dropout_rate),
                               nn.Linear(input_dim, hidden_dim),
                               nn.BatchNorm1d(hidden_dim),
                               nn.LeakyReLU(0.1)
                               )

        if self.head == 'ml':
            self.ml_decoder_head = MLDecoder(num_class, initial_num_features=input_dim, dim_feedforward=1024) # initilization
        else:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=number_layers, bidirectional=False, dropout=dropout_rate)
        # if self.bidirectional:
        #     self.output = nn.Sequential(nn.Linear(hidden_dim*2*num_class, self.num_class),
        #                             nn.Sigmoid())
        # else:
            self.output = nn.Sequential(nn.Linear(hidden_dim*num_class, self.num_class),
                                nn.Sigmoid())

    def forward(self, image_features, det_features, cap_features):

        if len(image_features.shape) == 4:
            cap_features = cap_features.unsqueeze(-1).unsqueeze(-1)
            cap_features = cap_features.expand(-1,-1,image_features.shape[-2], image_features.shape[-1])
           

        x = torch.cat((image_features, cap_features), dim=1).detach()


        # ml decoder
        if self.head == 'ml':
            if len(x.shape) == 2:
                x = x.unsqueeze(-1).unsqueeze(-1)
            x = F.dropout(x, 0.3)
            output = self.ml_decoder_head(x)
            output = torch.sigmoid(output)

        # image_features = self.backbone.features(x)
        # image_features = F.adaptive_avg_pool2d(image_features, 1)

        else:
            if len(x.shape) == 4:
                x = F.adaptive_avg_pool2d(x, 1).squeeze()
            image_features = self.fc(x)
            # image_features [batch_size, feature_size]
            image_features = image_features.view(image_features.shape[0], 1, -1)
            # image_features [batch_size, seq_len(num_class), feature_size]
            image_features = image_features.expand(image_features.shape[0], self.num_class, image_features.shape[-1])
            #print(image_features.shape)
            # output [batch_size, seq_len, num_directions * hidden_size]
            output, (_, _) = self.lstm(image_features)
            # output [batch_size, seq_len * num_directions * hidden_size]
            output = output.reshape(output.shape[0], -1)
            output = self.output(output)

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