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
    train_data_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_data_loader = data.DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)
    return train_data_loader, test_data_loader

train,test,scaler = train_test_split_and_normalize(import_data_for_training())
train_dl, test_dl = prepare_dataloaders(train, test)
print(train_dl)