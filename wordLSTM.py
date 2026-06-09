import torch
from torch import nn
from torch.utils.data import DataLoader
import csv
import numpy.typing as npt
import numpy as np
from typing import Any
import trinketbox.ai.utils.tokenDataset as integerDataset

from griot import word as griotNPW
from griot import tool




### LSTM Architecture Parameters TODO: pick snakecase or camelcase smh
inSize : int = 384         # Context window
outSize : int = 1          # How many words to predict
embedding_dim : int = 256  # Embedding dimension for vocabulary
hidden_size : int = 768    # Hidden size for each LSTM layer
num_layers : int = 3       # Number of LSTM layers
dropout : float = 0.35     # Dropout for regularization between LSTM layers
device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPath = 'model.pth'
vocab : griotNPW.StrictVocab 


class NeuralNetwork(nn.Module):
    def __init__(self, vocSize, inSize, outSize, 
                 embedding_dim=128, 
                 hidden_size=256, 
                 num_layers=2, 
                 dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocSize, embedding_dim,padding_idx=vocab.padTok[0],scale_grad_by_freq=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.layerNorm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, outSize * vocSize)
        self.softMax = nn.Sigmoid()
        self.outSize = outSize
        self.vocSize = vocSize
        
    def forward(self, x):
        # x shape: (batch_size, inSize)
        x = x.to(self.embedding.weight.device)
        x = self.embedding(x)  # (batch_size, inSize, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, inSize, hidden_size)
        # Use the last timestep output
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        x = self.layerNorm(x)
        logits = self.linear(x)  # (batch_size, outSize * vocSize)
        #logits = self.softMax(logits)
        logits = logits.view(-1, self.outSize, self.vocSize)  # (batch_size, outSize, vocSize)
        return logits

    def saveWeights(self,weightsPath:str='model.pth') -> None:
        torch.save(self.state_dict(),weightsPath)
        return
    def loadWeights(self,weightsPath:str='model.pth') -> None:
        print('attempting to load weights')
        try:
            with open(weightsPath,'rb') as fi:
                self.load_state_dict(torch.load(fi),strict=False)
        except:
            print('load failed')
        return



def create() -> NeuralNetwork:
    print('creating model,,,')
    model = NeuralNetwork(vocSize=len(vocab), inSize=inSize, outSize=outSize, 
                            embedding_dim=embedding_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout).to(device)
    return model