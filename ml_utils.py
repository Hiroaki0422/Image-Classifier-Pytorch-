# Imports here
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch import nn
from collections import OrderedDict
import torch.optim as optim
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import json

def getNameMap(category_names='./cat_to_name.json'):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def getModel(arc='vgg19', gpu=True, hidden_units=4096):
    if arc == 'vgg19':
        mymodel = models.vgg19(pretrained=True)
    elif arc == 'vgg16':
        mymodel = models.vgg16(pretrained=True)
    elif arc == 'vgg13':
        mymodel = models.vgg13(pretrained=True)
    else:
        print('that model is not available, using vgg19')
        mymodel = models.vgg19(pretrained=True)
    
    for param in mymodel.parameters():
        param.requires_grad_(False)
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    mymodel = mymodel.to(device)
    myclassifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('drop', nn.Dropout()),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    myclassifier = myclassifier.to(device)
    mymodel.classifier = myclassifier
    return mymodel, device

def validation(model, val_loader):
    running_loss = 0.0
    print('.....running on validation set')
    output_list = []
    labels_list = []
    for i, data in enumerate(val_loader):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = mymodel(inputs)
        loss = criterion(outputs, labels)
        
        output_labels = outputs.argmax(1)
        output_list.append(output_labels)
        labels_list.append(labels)

        # print statistics
        running_loss += loss.item()
    print('validation loss: ', running_loss / (i + 1))
    output_labels = torch.cat(output_list)
    val_labels = torch.cat(labels_list)
    print('validation accuracy:', accuracy_score(output_labels.cpu(), val_labels.cpu()))
    
def save_checkpoint(model, hidden_units, path='./models'):
    save_dict = {}
    save_dict['model'] = model.state_dict()
    save_dict['hidden_units'] = hidden_units
    path = path + '/mymodel.pth'
    torch.save(save_dict, path)
    print('model saved')
    
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = models.vgg19(pretrained=False)
    hidden_units = checkpoint['hidden_units']
    myclassifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('drop', nn.Dropout()),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = myclassifier
    model.load_state_dict(checkpoint['model'])
    return model 

def loadData(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        # The validation and testing sets are used to measure the model's performance
        ## on data it hasn't seen yet. For this you don't want any scaling or
        ## rotation transformations,but you'll need to resize then crop the images to the appropriate size.

        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True),
        'validation': torch.utils.data.DataLoader(val_datasets, batch_size=32),
        'test': torch.utils.data.DataLoader(test_datasets, batch_size=32)
    }
    
    return dataloaders

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    img = transformer(img).float()
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img.transpose(1,2,0) - mean)/std
    img = img.transpose(2,0,1)
    return img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    test_img = process_image(image_path)
    test_img = torch.from_numpy(np.array([test_img])).float()
    
    results = model.forward(test_img)
    probs = torch.exp(results)
    
    topkprobs, topklabs = probs.topk(topk)
    topkprobs = topkprobs.detach().numpy()
    topklabs = topklabs.detach().numpy()
    
    return topkprobs[0].tolist(), topklabs[0].tolist()

