import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import torch.utils.data as data
import numpy as np

def import_data_for_training():
    df = pd.read_csv(Path(__file__).parent / "prepared_data.csv")
    df = df.drop(df.columns[0], axis=1)
    return df

def train_test_split_and_normalize(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    train = data.sample(frac=0.85, random_state=20)
    test = data.drop(train.index)
    
    train_features = train.iloc[:, 1:].values
    train_targets = train.iloc[:, 0].values.reshape(-1, 1)  # Keep targets as a 2D array
    test_features = test.iloc[:, 1:].values
    test_targets = test.iloc[:, 0].values.reshape(-1, 1)    # Keep targets as a 2D array

    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    train_data = np.hstack((train_targets, train_features))
    test_data = np.hstack((test_targets, test_features))
    
    return train_data, test_data, scaler


def prepare_dataloaders(train, test):
    train_dataset = data.TensorDataset(torch.from_numpy(train[:, 1:]).float(), torch.from_numpy(train[:, 0]).float())
    test_dataset = data.TensorDataset(torch.from_numpy(test[:, 1:]).float(), torch.from_numpy(test[:, 0]).float())
    train_data_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_data_loader, test_data_loader

class PeopleCounterLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1, num_layers=1):
        super(PeopleCounterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(64, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        
        # REGRESSION:
        out = self.fc(out)
        out = torch.sigmoid(out) * 5
        out = torch.round(out)
        return out

        # CLASSIFIER SOFTMAX:
        # out = self.fc(out)       
        # out = torch.softmax(out, dim=1)  # Softmax over the class dimension
        # return out

def train_reg(model, train_loader, test_loader, lr=0.001, num_epochs=10, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.unsqueeze(1).to(device)
                y_val = y_val.unsqueeze(1).to(device)
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

from sklearn.metrics import mean_squared_error, mean_absolute_error

def test_regression_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad(): 
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)

    good = 0

    for i in range(len(all_predictions)):
        print("pred: ", float((all_predictions[i])), " | target: ", float(all_targets[i]))
        if round(float((all_predictions[i]))) == round(float(all_targets[i])):
            good = good + 1

    print(f"accuracy: {good/len(all_predictions)}")
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    return mse, mae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_d, test, scaler = train_test_split_and_normalize(import_data_for_training())
train_dl, test_dl = prepare_dataloaders(train_d, test)

model = PeopleCounterLSTM(input_size=10, hidden_size=64, output_size=1, num_layers=1)
train_reg(model, train_dl, test_dl, lr=0.001, num_epochs=2500, device=device)
test_regression_model(model, test_dl, device=device)

model_save_path = Path(__file__).parent / "trained_people_counter_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")