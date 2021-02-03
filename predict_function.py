# Programmer: Oscar Marklund
# Submission date: 03-Feb-2021
# Last altered: 03-Feb-2021

def loadandpredict(ch_path, image_path, topk, cat_names, gpu):
    # Receives user input with defaults in case to load a trained model architecture on images of flowers, pitch an image to it to be classified and return results
    
    #Imports
    import numpy as np
    import torch.nn.functional as F

    import argparse
    import json
    import time
    import torch
    from torch import nn, optim
    from torchvision import datasets, transforms, models
    import os
    from collections import OrderedDict
    import PIL
    from PIL import Image

    # GPU or CPU is made the device of choice
    # USER INPUT TURNS GPU ON OR OFF
    if gpu is True:
        if torch.cuda.is_available() is True:
            device = torch.device('cuda')
            print('GPU will gladly inference')
        else:
            device = torch.device('cpu')
            print('GPU was not found so CPU is being used')
    else:
        device = torch.device('cpu')
        print('Inferencing will commence on CPU')

    # Overcomplicated loading function dependant on architecture of checkpoint's model and whether the GPU is activated or not
    def load_checkpoint(path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location = 'cpu')

        vgg11 = 'vgg11'
        vgg13 = 'vgg13'
        vgg16 = 'vgg16'
        vgg19 = 'vgg19'

        model_architecture = checkpoint['model']
        if model_architecture == vgg11:
            model = models.vgg11(pretrained=True)
        elif model_architecture == vgg13:
            model = models.vgg13(pretrained=True)
        elif model_architecture == vgg16:
            model = models.vgg16(pretrained=True)
        elif model_architecture == vgg19:
            model = models.vgg16(pretrained=True)
        
        model.class_to_idx =(checkpoint['class_to_idx'])
        model.classifier = (checkpoint['classifier'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    model = load_checkpoint(ch_path)


    # Inferencing occurs
    # Process image via transforms
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''  
        image = Image.open(image)

        preprocess = transforms.Compose([transforms.Resize(255), #255x255
                                         transforms.CenterCrop(224), #cropped centrally to 224x224
                                         transforms.ToTensor(), #makes tensor and transposition is not necessary for it is not an np.array
                                         transforms.Normalize([0.485, 0.456, 0.406], #normalises via mean
                                                              [0.229, 0.224, 0.225])]) #normalises via std deviations

        image = preprocess(image)
        return image

    #process_image('/home/workspace/ImageClassifier/flowers/test/21/image_06805.jpg')
    # USER INPUTS IMAGE OF THEIR CHOICE
    process_image(image_path)


    # Class prediction function to return probs and classes
    def predict(image_path, model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model with USER CHOICE of category names.
        '''

        model.to(device)
        image = process_image(image_path)
        image = image.to(device)
        image_classes_dict = {v: k for k, v in model.class_to_idx.items()}
        model.eval()

        with torch.no_grad():
            image.unsqueeze_(0)
            output = model.forward(image)
            ps = torch.exp(output)
            probs, classes = ps.topk(topk)
            probs, classes = probs[0].tolist(), classes[0].tolist()

            return_classes = []
            for c in classes:
                return_classes.append(image_classes_dict[c])

            return probs, return_classes

    # THIS IS WHERE USER'S TOPK INPUT IS REQUIRED
    probs, classes = predict(image_path, model, topk)


    # A comprehensive dictionary of actual flower names with their associated probabilities is made
    # USER DECIDES JSON categorising FILE  
    flower_classes = []
    probs_percentages = []
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    for x in classes:
        flower_classes.append(cat_to_name[x].title())
    flower_classes.reverse()

    for x in probs:
        probs_percentages.append("{:.0%}".format(x))

    # Merging flowers and their percetnages
    zip_flowersandperc = zip(flower_classes, probs_percentages)
    flower_percentages = dict(zip_flowersandperc)

    
    # Final results are displayed
    print()
    print("The classifier deems your image to be a: " + flower_classes[0])
    print("The top results were...")

    for f in flower_percentages: 
        #print("Flower: " + flower_classes[x] + "Probability: " + probs_percentages[x])
        print("Flower: {:25}   Probability: {}".format(f, flower_percentages[f]))