import torch
import torch.nn as nn
import time

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 10
hidden_size = 50
output_size = 1
num_layers = 2
learning_rate = 0.001

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

#   INFERENCE

X_test = torch.randn(1, 20, input_size)

begin = time.time()

with torch.no_grad():
    prediction = model(X_test)
    print(f'Prediction: {prediction.item():.4f}')

print(f"Time of inference: {round(1000*(time.time()-begin))} ms")