import torch
from torchvision import models, transforms
import torch.nn as nn
from skimage import io, transform
from PIL import Image
from torch.autograd import Variable

#Transforms required to preproccess data for input to model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

#Set CPU as tensor proccessor
device = torch.device('cpu')

#Load resnet18
model = models.resnet18()

#Restructure network to output 2 features instead of 1000
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

#Load trained model
model.load_state_dict(torch.load("./model1.pt", map_location=device))

#Set model to evaluation phase
model.eval()

#function that returns 0 if a fire image is detected, 1 otherwise
def predict_image(image):
    #convert image to to PIL image
    image = Image.fromarray(image)
    #convert image to tensor and apply transforms
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    #load image to device
    input = input.to(device)
    #output tensor of probabs
    output = model(input)
    #prob with max value will be the class thats predicted
    index = output.data.cpu().numpy().argmax()
    return index