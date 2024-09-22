import torch
import torch.nn as nn
import torch.optim as optim
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

with torch.no_grad():
    prediction = model(X_test)
    print(f'Initial Prediction: {prediction.item():.4f}')

real_data = torch.randn(1, output_size) #simulation

#   FINE TUNING

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_finetune_epochs = 5

model.train()
for epoch in range(num_finetune_epochs):
    outputs = model(X_test)
    optimizer.zero_grad()
    loss = criterion(outputs, real_data)
    loss.backward()
    optimizer.step()
    print(f'Fine-tune Epoch [{epoch+1}/{num_finetune_epochs}], Loss: {loss.item():.4f}')


#   CONTINUOUS FINE TUNING

for step in range(10):
    with torch.no_grad():
        prediction = model(X_test)
        print(f'Prediction at step {step}: {prediction.item():.4f}')
    time.sleep(5)
    real_data = torch.randn(1, output_size)
    model.train()
    for epoch in range(num_finetune_epochs):
        outputs = model(X_test)
        optimizer.zero_grad()
        loss = criterion(outputs, real_data)
        loss.backward()
        optimizer.step()
        print(f'Fine-tune Epoch [{epoch+1}/{num_finetune_epochs}], Loss: {loss.item():.4f}')
