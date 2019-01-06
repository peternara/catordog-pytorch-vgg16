import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet50

from torch.autograd import Variable 
import pandas as pd

transform = T.Compose([
    T.Resize(224),  
    T.CenterCrop(224),  
    T.ToTensor(),  
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])



class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        '''
        if 'dog' in img_path.split('/')[-1] :
            label = 0
        else :
            label = int(img_path.split('/')[-1].split('.')[0])
        '''
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

#model = resnet50(pretrained=True)

model = vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096, 4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096, 2))

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

model = model.cuda()
cost = torch.nn.CrossEntropyLoss()
cost = cost.cuda()
#optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.classifier.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model, device_ids=[0,1])
model.to(device)


def train():
    epoch_num = 30
    batch_size = 150

    dataset = DogCat('/home/wumingjie/Dataset/catvsdog3/train_data', transforms=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i in range(epoch_num):
        running_loss = 0.0
        print('-----epoch', i, '-----')
        model.train = True
        for num, image in enumerate(dataloader):
            x_train, y_train = image
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            optimizer.zero_grad()
            output = model(x_train)
            loss = cost(output, y_train)
            print(num*batch_size, '/ 33000', 'loss:', loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('-----------Epoch:', i, ', loss', running_loss, '-----------')
        torch.save(model, '../log/dog_cat_model_vgg16.pkl')

def test():
    model = torch.load('../log/dog_cat_model_vgg16.pkl')
    #model = model.cuda()
    model.train = False
    #model.eval()
    test_dataset = DogCat('/home/wumingjie/Dataset/catvsdog3/test_data/', transforms=transform)
    test_dataloader = data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    eq = 0
    #print test_dataset
    for num, image in enumerate(test_dataloader):
        x_train, y_train = image
        x_train = Variable(x_train.cuda())
        y_train = y_train.cuda()
        y_pred = model(x_train)
        _, pred = torch.max(y_pred.data, 1)
        print(pred)
        #print(y_train)
        for i in range(50):
            if pred[i] == y_train[i]:
                eq = eq + 1
    print 'test_result', eq / 500.0

def predict():
    model = torch.load('../log/dog_cat_model_vgg16.pkl')
    model = model.cuda()
    #model.eval()
    test_dataset2 = DogCat('/home/wumingjie/Dataset/catvsdog/test_data/', transforms=transform)
    test_dataloader2 = data.DataLoader(test_dataset2, batch_size=100, shuffle=False)

    predict_id = []
    predict_label = []

    for num, image in enumerate(test_dataloader2):
        x_train, y_train = image
        x_train = Variable(x_train.cuda())
        y_train = Variable(y_train.cuda())

        print y_train
        #print x_train
        y_pred = model(x_train)
        _, pred = torch.max(y_pred.data, 1)
        print pred

        predict_id = predict_id + list(y_train.cpu().numpy())
        predict_label = predict_label + list(pred.cpu().numpy())
        #break

    print predict_label
    catordog = []
    for label in predict_label :
        if label == 0:
            catordog.append('Dog')
        else :
            catordog.append('Cat')

    print('save result...')

    lr_output = pd.DataFrame(data=predict_id, columns=['id'])
    lr_output['label'] = catordog
    lr_output = lr_output[['id', 'label']]
    lr_output.to_csv('submission.csv', index=False)

    print('finish.')


train()
#test()
#predict()
