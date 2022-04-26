import torch.nn as nn
import torch.nn.functional as F
import torch


class Caption(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True, dropout_rate=0, num_class=19, body='lstm', sigmoid=True):
        super(Caption, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bidirectional = bidirectional
        self.num_dir = 2 if self.bidirectional else 1
        self.body = body
        self.sigmoid = sigmoid

        if self.body == 'lstm':        
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_rate)
            if self.bidirectional:
                self.output = nn.Sequential(nn.Linear(hidden_dim*num_class*2, self.num_class), 
                                        nn.Sigmoid() if sigmoid else nn.Identity())
            else:
                self.output = nn.Sequential(nn.Linear(hidden_dim*num_class, self.num_class),
                                        nn.Sigmoid() if sigmoid else nn.Identity())
        else:
            encoder_layer = nn.TransformerEncoderLayer(input_dim, nhead=5, dim_feedforward=1024, dropout=0.2, batch_first=True)
            self.tf = nn.TransformerEncoder(encoder_layer, 1)
            #self.tf = nn.Transformer(d_model=input_dim, nhead=5, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=512, batch_first=True)
            self.output = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(5700, self.num_class),
                                        nn.Sigmoid() if sigmoid else nn.Identity())
            self.num_dir = 1 
        self.label_lstm = nn.LSTM(hidden_dim*self.num_dir, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_rate)
        

    def forward(self, x):
        input_lens = torch.LongTensor(list(map(len, x))).to('cuda:0')
        # padded inputs [batch_size, max_seq_len, embedding_dim]
        padded_inputs = torch.zeros(len(x), input_lens.max(), self.input_dim).to('cuda:0')
        for i, text in enumerate(x):
            text_len = input_lens[i]
            padded_inputs[i, :text_len] = torch.stack(text)
        
        if self.body == 'lstm':
            # packed inputs
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(padded_inputs,
                                                                    input_lens.cpu().numpy(),
                                                                    batch_first=True,
                                                                    enforce_sorted=False).to('cuda:0')
            
            _, (hidden, _) = self.lstm(packed_inputs)
            if self.bidirectional:
                output = torch.cat((hidden[0,:,:], hidden[1,:,:]), 1)
            else:
                output = hidden[0, :, :]
            # output [batch_size, hidden_dim*num_direction] -> [batch_size, num_class, hidden_dim*num_direction]
        else: # transformer
            
            output = self.tf(padded_inputs)
            output = output.mean(dim=1, keepdim=False)
            #print(padded_inputs.shape, output.shape)
        #print(output.shape, '??')
        #features = output
        output = output.reshape(output.shape[0], 1, -1)
        output = output.expand(output.shape[0], self.num_class, output.shape[-1])
        output, (_, _) = self.label_lstm(output)
        # output [batch_size, seq_len, num_direction * hidden_dim]
        output = output.reshape(output.shape[0], -1)


        #output = F.dropout(output, 0.5)
        output = self.output(output)

        #print(output.shape)
        if not self.sigmoid:
            predictions = torch.sigmoid(output) > 0.5
        else:
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