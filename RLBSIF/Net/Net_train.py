import numpy as np
import torch
import os, glob
import random, csv
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from torchvision import models
import torch.nn as nn
from Net import Net, BasicBlock
import warnings

warnings.filterwarnings("ignore")  
torch.manual_seed("")   
np.random.seed("")
total_epochs = 100
learning_rate = 0.02  #0.03
num_workers = 8
batch_size = 32

train_on_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    ])


class xiaofenzi_Dataset(Dataset):
    def __init__(self, dataset_dir, dataset_lable_dir):
        self.datadir = dataset_dir
        self.labledir = dataset_lable_dir
        self.path = os.path.join(self.datadir,self.labledir)
        self.npy_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, item):
        npy_name = self.npy_path[item]
        npy_item_path = os.path.join(self.datadir, self.labledir, npy_name)
        data = np.load(npy_item_path, allow_pickle=True)
        lable = int(self.labledir)
        return self.transform(data), torch.LongTensor([lable])
    def __len__(self):
        return len(self.npy_path)



#dataset_dir = ''

#dataset_dir = ''
#dataset_dir = ''
#dataset_dir = ''
#dataset_dir = ''
dataset_dir = ''

dataset_lable_ = 'positive'
dataset_lable_ = 'negative'
postive_data_set = xiaofenzi_Dataset()
negative_data_set = xiaofenzi_Dataset()
indices = list(range(len(dataset)))
np.random.shuffle(indices)  
train_idx = indices[: int(len(dataset)*0.8)]
valid_idx = indices[int(len(dataset)*0.8): int(len(dataset)*0.9)]
test_idx = indices[int(len(dataset)*0.9):]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
classes = ['disbinding_site', 'binding_site']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RMSIF_NET(BasicBlock=BasicBlock)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
valid_loss_min = np.Inf  
save_train_loss = []
save_Valid_loss = []

if __name__ == '__main__':
    for epoch in range(total_epochs):
        if epoch > 30:
            learning_rate = 0.01
        if epoch > 90:
            learning_rate = 0.001
        if epoch > 200:
            learning_rate = 0.0001
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for step, (data, target) in enumerate(train_loader):
            #data, target = data.cuda(), target.cuda().float()  #.unsqueeze(1)
            target = target.float()

            #print(len(data))
            #print(data.shape)
            #print(target)
            optimizer.zero_grad()  
            output = model(data)  
            #print(output)
            #print(output)

            loss = F.binary_cross_entropy(output,target)

            loss.backward() 
            optimizer.step()  
            train_loss += loss.item() * data.size(0) 

        model.eval()  
        for step, (data, target) in enumerate(valid_loader):
            #data, target = data.cuda(), target.cuda().float()   #.unsqueeze(1)
            target = target.float()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)  
            valid_loss += loss.item() * data.size(0)  


        train_loss = train_loss / len(train_loader.dataset)  
        valid_loss = valid_loss / len(valid_loader.dataset)

        save_train_loss.append(train_loss)
        save_Valid_loss.append(valid_loss)


        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), '')
            valid_loss_min = valid_loss
 

    test_loss = 0.0  
    class_correct = list(0. for i in range(2))  
    class_total = list(0. for i in range(2))  


    model.eval()  
    state_dict = torch.load()
    model.load_state_dict(state_dict)
    for step, (data, target) in enumerate(test_loader):
        #data, target = data.cuda(), target.cuda().float()
        target = target.float()
        output = model(data)
        # target1 = target.float().unsqueeze(1)
        loss = F.binary_cross_entropy(output, target)
        test_loss += loss.item() * data.size(0)

        #pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).cuda()
        pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output])

        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
            correct_tensor.cpu().numpy())  

        for i in range(output.shape[0]):  
            label = target.data[i] 
            label = label.int()
            class_correct[label] += correct[i].item()  
            class_total[label] += 1
    test_loss = test_loss / len(test_loader.dataset)  
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))  
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    #np.save("loss/save_train_loss.npy", save_train_loss)
    #np.save("loss/save_Valid_loss.npy", save_Valid_loss)


