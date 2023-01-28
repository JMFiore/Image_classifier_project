import argparse
import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt 
import os

## This script alows you to train two different types of neaural networks inception_v3 by default or VGG.
#they have diferent image shape inputs and characteristics. Hope you enjoy the scprit and be uderstandable,cheers!


#Define our parser
parser = argparse.ArgumentParser(description= "Neural Network Train function")

#Define our arguments 
parser.add_argument('data_directory', help='specify you data directory where there are 3 folders trainig, validation and testing', type=str, default='/home/workspace/ImageClassifier/flowers')
parser.add_argument('--save_dir', help='directory path to save the model', default='/home/workspace/ImageClassifier', type=str)
parser.add_argument('--arch', help= 'indicate the CNN architecture to use. inception_v3 or vgg16. def inception', default='inception_v3', type=str)
parser.add_argument('--batch', help='specify the batch number, less batch slower pc can run it but takes more time default 64', type=int,default=64)
parser.add_argument('--learning_rate' , help='type the learning rate at witch you NN is going to train, default lr=0.001',default=0.001,type=float)
parser.add_argument('--hidden_units', help='Choose the number of hidden neurons in the hidden layer',type=int, default=540)
parser.add_argument('--epochs', help='Select the number of epochs to train your NN',type=int, default=20)
parser.add_argument('--gpu', help='True for use the GPU or False for use CPU', type=bool, default=True)

#Parse arguments
args=parser.parse_args()

def get_data(data_dir = args.data_directory,arch=args.arch):
    ''' Recives a directory where there are 3 folders:train, test, validation. Preparate the data for the model. returns 4 elements
    traindataloader, testdataloader, valdataloader and Train_data '''
    ## Finds the folders for train ,test and validation. In the path porvided
    tr=0
    te=0
    va=0
    for folder in os.listdir(data_dir):
        tr+=1
        te+=1
        va+=1
        if ('tr' in str(folder).lower()):
            train_dir=data_dir + '/' + str(folder)
            tr-=1
        if tr >= 3:
            print('Training folder is not in the directory specified')

        if 'te' in str(folder).lower():
            test_dir=data_dir + '/' + str(folder)
            te-=1
        if te >= 3:
            print('Testing folder is not in the directory specified')
        if 'va' in str(folder).lower():
            val_dir=data_dir + '/' + str(folder)
            va-=1
        if va >=3:
            print('Validation folder is not in the directory specified')

    #Defines the shape of the image imput for the model 
    if arch == 'inception_v3':
        Resize = [299,299]
    else:
        Resize = [224,224]

    #Defines the transformation for each of the dataset to use        
    train_transformer=transforms.Compose([transforms.CenterCrop([300,300]),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.Resize(Resize),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    data_transformer = transforms.Compose([transforms.CenterCrop([300,300]),
                                           transforms.Resize(Resize),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                          ])

    Train_data = datasets.ImageFolder(train_dir, transform= train_transformer)

    if test_dir != type(None): 
        Test_data = datasets.ImageFolder(test_dir, transform=data_transformer)
        print('Test_data created succesfuly!')
    else: 
        print('No test data found')
    if val_dir != type(None) :
        Val_data = datasets.ImageFolder(val_dir, transform=data_transformer)
    else:
        print('No Validation data found')

    traindataloader= torch.utils.data.DataLoader(Train_data, batch_size=args.batch, shuffle=True)
    testdataloader=torch.utils.data.DataLoader(Test_data, batch_size=args.batch)
    valdataloader= torch.utils.data.DataLoader(Val_data,batch_size=args.batch)
    
    return traindataloader, testdataloader, valdataloader,Train_data     


def model_generation(arch = args.arch, hidden_units=args.hidden_units):
    ''' Takes the model arg and the hidden_units agr, imports the CNN and creates a new classifier returns a model ready to classify the flowes dataset'''
    
    model = getattr(torchvision.models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    print('Model created {}, starting to generate the classifier'.format(args.arch))
    print('Number of hidden units is: {}'.format(args.hidden_units))
    if arch == 'inception_v3':
        fc = nn.Sequential(nn.Linear(2048, hidden_units),
                       nn.Dropout(p=0.4),
                       nn.ReLU(),
                       nn.Linear(hidden_units,102),
                       nn.LogSoftmax(dim=1)
                       )
        model.fc=fc
        print('clasifier for Inception NN crated!')
        return model,fc
    else:
        classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                       nn.Dropout(p=0.4),
                       nn.ReLU(),
                       nn.Linear(args.hidden_units,102),
                       nn.LogSoftmax(dim=1)
                       )
    
        model.classifier=classifier
        print('classifier for VGG NN created!')
        return model, classifier



def train_the_model( model, traindataloader, valdataloader, gpu = args.gpu ,arch=args.arch, learning_rate= args.learning_rate, epochs = args.epochs):
    ''' Gets the model and trains it due to specifications, returns a trained model'''
    
    criterion = nn.NLLLoss()
    if arch == 'inception_v3':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    if (gpu == True and torch.cuda.is_available()):
        device = 'cuda'
        model.to(device)
    else:
        device = 'cpu'
        
    model.to(device)
    ## Left to add an option to return the array of the losses in a future
    #training_losses, testing_losses=[],[]
    print('Training is Initializating')
    for e in range(epochs):
        training_loss = 0
        for images, labels in traindataloader:
            images, labels = images.to(device), labels.to(device)
            model.train()
            Log_prob = model(images)
            if arch == 'inception_v3':
                Loss = criterion(Log_prob[0],labels)
            else:
                Loss = criterion(Log_prob, labels)
            training_loss+= Loss.item()
            Loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            
        else:
            with torch.no_grad(): 
                model.eval()
                testing_loss=0
                accuracy = 0
                for images, labels in valdataloader:
                    images, labels = images.to(device), labels.to(device)
                    Log_loss= model.forward(images)
                    Loss = criterion(Log_loss, labels)
                    testing_loss += Loss.item()

                    prob = torch.exp(Log_loss)
                    top_prob, top_class = prob.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    #training_losses.append(training_loss/len(traindataloader))
                    #testing_losses.append(testing_loss/ len(valdataloader))
        
        print('==== Epoch {}/{} ===='.format(e+1,epochs))
        print('Training Loss {:.5}'.format(training_loss/len(traindataloader)))
        print('Validation Loss {:.5}'.format(testing_loss/len(valdataloader)))
        print('Accuracy {:.4}'.format(accuracy/len(valdataloader)))
        
    return model , optimizer                                        

def save_model( model,classifier, optimizer , Train_data ,arch=args.arch): 
    '''Takes the characteristics of the model and saves it in a file .pth in the specified path '''
    if arch == 'inception_v3':
        model.class_to_idx = Train_data.class_to_idx
        model_dict = {'network': 'inception_v3',
                  'input_size_c': 2048,
                  'output_size': 102,
                  'learning_rate': args.learning_rate,       
                  'batch_size':args.batch,
                  'classifier' :classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
        torch.save(model_dict, args.save_dir + '/checkpoint_' + args.arch + '.pth')
        print('Model saved')
    else:
        model.class_to_idx = Train_data.class_to_idx
        model_dict = {'network': args.arch,
                      'input_size_c': 25088,
                      'output_size': 102,
                      'learning_rate': args.learning_rate,       
                      'batch_size':args.batch,
                      'classifier' : classifier,
                      'epochs': args.epochs,
                      'optimizer': optimizer.state_dict(),
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx}
        torch.save(model_dict, args.save_dir + '/checkpoint_' + args.arch + '.pth')
        print('Model saved')
  


def main():          
    traindataloader,testdataloader,valdataloader,Train_data = get_data()
    model, classifier= model_generation()
    train_model, optimizer =train_the_model(model=model, traindataloader=traindataloader, valdataloader=valdataloader)
    save_model(model=model, classifier=classifier, optimizer=optimizer, Train_data=Train_data)
    
if __name__ == "__main__":
    main()