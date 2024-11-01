import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import torch.utils.data as data

def import_data_for_training():
    df = pd.read_csv(Path(__file__).parent / "prepared_data.csv")
    df = df.drop(df.columns[0], axis=1)
    return df


def train_test_split_and_normalize(data):
    from sklearn.preprocessing import StandardScaler
    train=data.sample(frac=0.85,random_state=20)
    test=data.drop(train.index)
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train,test,scaler


def prepare_dataloaders(train, test):
    train_dataset = data.TensorDataset(torch.from_numpy(train[:,:-1]).float(),torch.from_numpy(train[:,-1]).float())
    test_dataset = data.TensorDataset(torch.from_numpy(test[:,:-1]).float(),torch.from_numpy(test[:,-1]).float())
    train_data_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_data_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    return train_data_loader, test_data_loader


class PeopleCounterLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1, num_layers=1):
        super(PeopleCounterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        
        out = self.fc(out)
        
        out = torch.sigmoid(out) * 5
        return out
    

def train(model, lr, num_epochs, X_train, Y_train, X_test, Y_test):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, Y_test)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')


train,test,scaler = train_test_split_and_normalize(import_data_for_training())
train_dl, test_dl = prepare_dataloaders(train, test)
print(train_dl)