import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() # neccesary!
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

