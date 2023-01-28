import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
import numpy as np 
import argparse
from PIL import Image
import json

#Parser
parser = argparse.ArgumentParser(description='Function that gets as imputs the path of an image, a model previusly trained and other options. Returns  the porbability of the image belong to a class and the name of a class')

#Arguments
parser.add_argument('--img_path', type=str, help='add a path of a image to be classified', default='/home/workspace/ImageClassifier/flowers/valid/1/image_06739.jpg')
parser.add_argument('--checkpoint', type=str, help='add the path of the chackpoint model', default='/home/workspace/ImageClassifier/checkpoint_inception_v3.pth')
parser.add_argument('--category_names', type=str, help= 'add the dictionary containing the names of the flowers and the number of the floder', default='/home/workspace/ImageClassifier/cat_to_name.json')
parser.add_argument('--gpu', type=bool, help='specify with True to enable gpu', default=True)
parser.add_argument('--top_k',type=int, help='choose the number of top values to get',default=3)

args=parser.parse_args()



def reconstruct_model(model_path= args.checkpoint ):
    '''Takes the model path arguments to load the data of the checkpoint, and rebuilds the model, returns a model and the size of the image that the model handles.'''
    
    print('Loading the model')
    model_data = torch.load(model_path)
    model= getattr(torchvision.models,model_data['network'])(pretrained=True)
    if (model_data['network'] == 'inception_v3'):
        fc = model_data['classifier']
        model.fc = fc
        Resize=[299,299]
    else:
        model.classifier= model_data['classifier']
        Resize=[224,224]
    model.optimizer= model_data['optimizer']
    model.state_dict= model_data['state_dict']
    model.epochs = model_data['epochs']
    model.class_to_idx = model_data['class_to_idx']
    print('Model loaded Succesfuly!')
    return model, Resize

def image_processing(Resize, image_path=args.img_path):
    print('Processing the image')
    image = Image.open(image_path)
    transformer = transforms.Compose([transforms.CenterCrop(299),
                                 transforms.Resize(Resize),
                                 transforms.ToTensor()])
    
    image_pil = transformer(image).float() 
    np_image = np.array(image_pil)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image


def model_predict(image,model, dict_path=args.category_names, gpu=args.gpu,k=args.top_k):
    '''Takes the image preprocessed and makes the predictions, then takes the classes of the model 
    makes one dictionary to pass the classes of the model to the classes of the flowers in numbers, then uses
    the dictionary that conains the names of the flowers and returns two arrays, of the probabilities and the other,the  names of the flowers in order with the probs.'''
    print('Generating predictions...')
    if gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.to(device)
    ready_image=torch.from_numpy(image).unsqueeze(dim=0).cuda().float()
    model.eval()
    with torch.no_grad():
        Log_output= model(ready_image)
        prob=torch.exp(Log_output)
        top_prob, top_class = prob.topk(k,dim=1)

    f_class={}
    for k, v in model.class_to_idx.items():
         f_class[str(v)]=str(k) 
    
    data= open(dict_path)
    flowers_dict = json.load(data)
    flowers_class=[]
    for number in np.array(top_class)[0]:
        flowers_class.append(flowers_dict[f_class[str(number)]])
    
    
    return np.array(top_prob)[0], flowers_class
    
def main():    
    model,Resize = reconstruct_model()  
    image= image_processing(Resize = Resize)
    top_pred, top_classes = model_predict(image,model)
    print('Probabilities:')
    print(top_pred)
    print('Classes:')
    print(top_classes)
if __name__ == '__main__':
    main()