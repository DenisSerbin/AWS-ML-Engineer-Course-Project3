#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import smdebug.pytorch as smd
import logging
import os
import sys
import json
import copy

#TODO: Import dependencies for Debugging and Profiling

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration for Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info(f"Testing Model on the test data set.")

    model.eval()
    hook.set_mode(smd.modes.EVAL) # Debugger hook mode = EVAL
    
    running_loss = 0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects.double() / len(test_loader.dataset)
        logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))

        
def train(model, train_loader, validation_loader, criterion, optimizer, device, hook, epochs = 20):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''    
    image_dataset = {'train' : train_loader, 'valid' : validation_loader}
    
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Train set length: {len(train_loader.dataset)}")
    logger.info(f"Validation set length: {len(validation_loader.dataset)}")
    
    best_loss = 1e6
    best_acc = 0
    loss_counter = 0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)

            running_loss = 0.0
            running_corrects = 0
            #running_samples = 0
            
            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #running_samples += len(inputs)
                
                #if running_samples % 2000  == 0:
                #   accuracy = running_corrects / running_samples
                #    logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                #        running_samples,
                #        len(image_dataset[phase].dataset),
                #        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                #        loss.item(),
                #        running_corrects,
                #        running_samples,
                #        100.0 * accuracy,
                #    ))

            epoch_loss = running_loss / len(image_dataset[phase].dataset)
            epoch_acc = running_corrects.double() / len(image_dataset[phase].dataset)
            
            logger.info('Phase: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(phase, epoch_loss, 100 * epoch_acc))
            
            #epoch_loss = running_loss / running_samples
            #epoch_acc = running_corrects / running_samples
            
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print('Training complete. Best validation accuracy: {:4f}'.format(100 * best_acc))
    model.load_state_dict(best_model_wts)
    
    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 512),
                             nn.Dropout(p = 0.2),
                             nn.ReLU(inplace = True),
                             nn.Linear(512, 256),
                             nn.Dropout(p = 0.2),
                             #nn.ReLU(inplace = True),
                             #nn.Dropout(p = 0.2),
                             #nn.Linear(512, 256),
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133)
                            )
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    train_path = os.path.join(data, "train/")
    val_path = os.path.join(data, "valid/")
    test_path = os.path.join(data, "test/")
    
    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(0, shear = 10, scale = (0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(degrees = 90),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
    train_set = torchvision.datasets.ImageFolder(root = train_path, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
    val_set = torchvision.datasets.ImageFolder(root = val_path, transform = test_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)

    test_set = torchvision.datasets.ImageFolder(root = test_path, transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    
    logger.info(f"Hyperparameters: Batch Size: {args.batch_size}, Learning rate: {args.lr}")
    
    model = net()
    model = model.to(device)

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    train_loader, val_loader, test_loader = create_data_loaders(args.train, args.batch_size)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    hook.register_loss(criterion)
    
    optimizer = optim.RMSprop(model.fc.parameters(), lr = args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model = train(model, train_loader, val_loader, criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    
    save_path = args.model_dir + "/model.pth"
    logger.info(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved!")
    #torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    #torch.save(model, os.path.join(args.model_dir, 'model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 0.002)
    
    # input data and model directories
    parser.add_argument('--model_dir', type = str, default = os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type = str, default = os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--train', type = str, default = os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type = str, default = os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--val', type = str, default = os.environ['SM_CHANNEL_VAL'])
    
    args = parser.parse_args()

    # ... load from args.train, args.test, and args.val to train a model, write model to args.model_dir.
    main(args)
