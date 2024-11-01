import torch
import torch.nn as nn

class UNMTDecoder(nn.module):
    def __init__(self, n_output_tokens:int, bert_pretrained_type: str = 'bert-base-multilingual-cased', input_dim: int = 768, hidden_dim:int = 1024, ):
        '''
            bert_pretrained_type: str, type of bert model to use
            output_dim: int, dimension of output of encoder
        '''
        super(UNMTDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_output_tokens = n_output_tokens

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.LSTM = nn.LSTM(self.input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_output_tokens)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, max_len:int, stop_token:int):
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size,)
        #not completed
        x, _ = self.LSTM(x)
        x = self.fc(x)
        x = self.relu(x)
        return x