import argparse
import ml_utils 
import torch.optim as optim 
from torch import nn

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', default="./flowers")
parser.add_argument('--arch', default="vgg19", type=str)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_units', default=4096, type=int)
parser.add_argument('--save_dir', default="./models")
parser.add_argument('--epochs', default=5, type=int)

# load datasets
args = parser.parse_args()
dataloaders = ml_utils.loadData(args.data_dir)

# load model
mymodel, device = ml_utils.getModel(args.arch, args.gpu, args.hidden_units)
print('model loaded')

# define optimizer & loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mymodel.classifier.parameters(), lr=args.learning_rate, momentum=0.9)

ml_utils.save_checkpoint(mymodel, args.hidden_units, args.save_dir)

# train model
print('training start')
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloaders['train'], 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mymodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 0:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
    ml_utils.validation(mymodel, dataloaders['validation'])

print('Finished Training')

# save model
ml_utils.save_checkpoint(mymodel, args.save_dir)

