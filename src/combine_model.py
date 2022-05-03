import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from ml_decoder import MLDecoder

class CombineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=36, num_class=19, bidirectional=True, dropout=0):
        super(CombineModel, self).__init__()
        self.num_class = num_class
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=bidirectional, dropout=0)
        self.output = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(hidden_dim*num_class*2, num_class),
                                    nn.Sigmoid(),  
                                    )

    def forward(self, features):

        image_features = torch.cat(features, 1)
        # if det_features is not None:
        #     image_features = torch.cat((image_features, det_features, cap_features), 1)
        # else:
        #     image_features = torch.cat((image_features, cap_features), 1)

        image_features = image_features.view(image_features.shape[0], 1, -1)
        # image_features [batch_size, seq_len(num_class), feature_size]
        image_features = image_features.expand(image_features.shape[0], self.num_class, image_features.shape[-1])
        #print(image_features.shape)
        # output [batch_size, seq_len, num_directions * hidden_size]
        output, (_, _) = self.lstm(image_features)
        output = output.reshape(output.shape[0], -1)

        output = self.output(output)
        predictions = output > 0.5
        return predictions, output

def evaluate(predictions, labels):
    if len(predictions)==0:
        return 0
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    samples_f1 = f1_score(labels, predictions, average='samples')
    return micro_f1, macro_f1, weighted_f1, samples_f1