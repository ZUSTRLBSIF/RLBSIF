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
from RMSIF_NET import RMSIF_NET, BasicBlock
import warnings
warnings.filterwarnings("ignore")  


torch.manual_seed()   
np.random.seed()
total_epochs = 10
learning_rate = 0.01  #0.03
num_workers = 8
batch_size = 32

train_on_gpu = torch.cuda.is_available()
transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.ToTensor(),
    ])

class testdata(Dataset):
    def __init__(self, testdatadir):
        self.datadir = testdatadir
        self.datalist = os.listdir(testdatadir)
        self.transform = transform
    def __getitem__(self,item):
        dataname = self.datalist[item]
        datapath = os.path.join(self.datadir, dataname)
        data = np.load(datapath ,allow_pickle = True)
        #order = int(self.datalist[item][0:1])
        order = int(str(self.datalist[item].split(".")[0]))
        return self.transform(data), torch.LongTensor([order])
    def __len__(self):
        return len(self.datalist)    

#testdatadir = ''
testdatadir = ''
data_set = testdata(testdatadir)
indices = list(range(len(data_set)))
#test_sampler = SubsetRandomSampler(indices)
#test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RMSIF_NET(BasicBlock=BasicBlock)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
valid_loss_min = np.Inf 
predict_per = {}


if __name__ == '__main__':

    test_loss = 0.0  
    class_correct = list(0. for i in range(2))  
    class_total = list(0. for i in range(2))  


    model.eval()  
    state_dict = torch.load('')
    model.load_state_dict(state_dict)
    for step, (data, order) in enumerate(test_loader):
        #data = data.cuda()
        order = order.int()
        output = model(data)
        print(output,order)

        for i in range(output.shape[0]):  
            predict_per[order[i].item()] = output[i].item() 
    np.save('', predict_per, allow_pickle=True)





