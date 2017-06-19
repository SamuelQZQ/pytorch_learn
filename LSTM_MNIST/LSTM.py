from torch import nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        '''
        x (batch, seq_len, 28)
        r_out (batch, seq_len, 64)
        out (batch, 10)
        '''
        r_out, (h_n, c_n) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])
        return out

