from torch import nn
import torch

class Discriminator(nn.Module):
    '''
        Discriminator model, operates on output of encoder
    '''
    def __init__(self, input_dim: int, num_langs: int = 3, dropout: float = 0.1):
        '''
            input_dim: int, dimension of input to discriminator
            num_langs: int, number of languages to classify
            dropout: float, dropout rate for discriminator
        '''
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(1024, num_langs),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__ == '__main__':
    model = Discriminator(1024)
    x = torch.randn(32, 1024)
    print(model(x).shape)