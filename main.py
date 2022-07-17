import streamlit as st 
import os

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
# from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

import keras.models.Model as Model
# from keras.layers import *
# from keras.optimizers import *
# from keras.utils import *
# from keras.callbacks import *

# from keras.applications.densenet import DenseNet121, preprocess_input

# import cv2

st.title("GUESS WHAT DOG YOU ARE")

st.write("""
# Put your picture and see what is your nearest breed!
""")
filename = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#      # To read file as bytes:
#      bytes_data = uploaded_file.getvalue()
#      st.write(bytes_data)

#      # To convert to a string based IO:
#      stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#      st.write(stringio)

#      # To read file as string:
#      string_data = stringio.read()
#      st.write(string_data)

#      # Can be used wherever a "file-like" object is accepted:
#      dataframe = pd.read_csv(uploaded_file)
#      st.write(dataframe)


# uploaded_file = st.file_uploader("Choose a Image")
# if uploaded_file is not None:
#     bytes_data = uploaded_file.getvalue()
#     image = Image.open(BytesIO(bytes_data)).convert("RGB")
#     img_for_plot = np.array(image)
    
#     img = transforms.ToTensor()(image)
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#     img = normalize(img).unsqueeze(dim=0)   
#     result = model(img).squeeze(dim=0)
#     predict_idx = result.argmax().item()
#     prob = torch.softmax(result, dim=0)
#     st.image(img_for_plot, use_column_width=True)
#     st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")


# pretrained_model.summary()
breed_list = ['n02093256-Staffordshire_bullterrier',
 'n02094114-Norfolk_terrier',
 'n02086079-Pekinese',
 'n02088364-beagle',
 'n02107312-miniature_pinscher',
 'n02113624-toy_poodle',
 'n02097130-giant_schnauzer',
 'n02095889-Sealyham_terrier',
 'n02101556-clumber',
 'n02113712-miniature_poodle',
 'n02108000-EntleBucher',
 'n02098105-soft-coated_wheaten_terrier',
 'n02110958-pug',
 'n02088094-Afghan_hound',
 'n02099849-Chesapeake_Bay_retriever',
 'n02111889-Samoyed',
 'n02109047-Great_Dane',
 'n02097474-Tibetan_terrier',
 'n02102973-Irish_water_spaniel',
 'n02094258-Norwich_terrier',
 'n02102480-Sussex_spaniel',
 'n02086646-Blenheim_spaniel',
 'n02097658-silky_terrier',
 'n02090379-redbone',
 'n02092339-Weimaraner',
 'n02091831-Saluki',
 'n02096294-Australian_terrier',
 'n02086240-Shih-Tzu',
 'n02108551-Tibetan_mastiff',
 'n02086910-papillon',
 'n02096177-cairn',
 'n02094433-Yorkshire_terrier',
 'n02110627-affenpinscher',
 'n02108422-bull_mastiff',
 'n02107908-Appenzeller',
 'n02096051-Airedale',
 'n02088466-bloodhound',
 'n02100735-English_setter',
 'n02106382-Bouvier_des_Flandres',
 'n02112018-Pomeranian',
 'n02096437-Dandie_Dinmont',
 'n02107574-Greater_Swiss_Mountain_dog',
 'n02105251-briard',
 'n02097047-miniature_schnauzer',
 'n02090721-Irish_wolfhound',
 'n02100877-Irish_setter',
 'n02100236-German_short-haired_pointer',
 'n02113799-standard_poodle',
 'n02113978-Mexican_hairless',
 'n02090622-borzoi',
 'n02101388-Brittany_spaniel',
 'n02106550-Rottweiler',
 'n02085936-Maltese_dog',
 'n02110806-basenji',
 'n02097209-standard_schnauzer',
 'n02105162-malinois',
 'n02106166-Border_collie',
 'n02104365-schipperke',
 'n02089078-black-and-tan_coonhound',
 'n02093991-Irish_terrier',
 'n02088632-bluetick',
 'n02098413-Lhasa',
 'n02091032-Italian_greyhound',
 'n02109961-Eskimo_dog',
 'n02111500-Great_Pyrenees',
 'n02097298-Scotch_terrier',
 'n02100583-vizsla',
 'n02107683-Bernese_mountain_dog',
 'n02089867-Walker_hound',
 'n02116738-African_hunting_dog',
 'n02099601-golden_retriever',
 'n02113186-Cardigan',
 'n02101006-Gordon_setter',
 'n02093754-Border_terrier',
 'n02115641-dingo',
 'n02098286-West_Highland_white_terrier',
 'n02110063-malamute',
 'n02099267-flat-coated_retriever',
 'n02099429-curly-coated_retriever',
 'n02091134-whippet',
 'n02105855-Shetland_sheepdog',
 'n02091467-Norwegian_elkhound',
 'n02093428-American_Staffordshire_terrier',
 'n02105505-komondor',
 'n02106662-German_shepherd',
 'n02105056-groenendael',
 'n02085782-Japanese_spaniel',
 'n02085620-Chihuahua',
 'n02112137-chow',
 'n02095570-Lakeland_terrier',
 'n02106030-collie',
 'n02093647-Bedlington_terrier',
 'n02108915-French_bulldog',
 'n02087046-toy_terrier',
 'n02102040-English_springer',
 'n02110185-Siberian_husky',
 'n02091244-Ibizan_hound',
 'n02102318-cocker_spaniel',
 'n02108089-boxer',
 'n02111129-Leonberg',
 'n02102177-Welsh_springer_spaniel',
 'n02111277-Newfoundland',
 'n02113023-Pembroke',
 'n02087394-Rhodesian_ridgeback',
 'n02091635-otterhound',
 'n02096585-Boston_bull',
 'n02093859-Kerry_blue_terrier',
 'n02099712-Labrador_retriever',
 'n02104029-kuvasz',
 'n02105412-kelpie',
 'n02115913-dhole',
 'n02092002-Scottish_deerhound',
 'n02095314-wire-haired_fox_terrier',
 'n02105641-Old_English_sheepdog',
 'n02088238-basset',
 'n02109525-Saint_Bernard',
 'n02112350-keeshond',
 'n02089973-English_foxhound',
 'n02107142-Doberman',
 'n02112706-Brabancon_griffon']


num_classes = len(breed_list)


label_maps = {}
label_maps_rev = {}

for i, v in enumerate(breed_list):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})

def create_functional_model():

    inp = Input((224, 224, 3))
    backbone = DenseNet121(input_tensor=inp,
                           weights="imagenet",
                           include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outp = Dense(num_classes, activation="softmax")(x)
#     model = Model(inp, outp)
    model = Model(inp, outp)
    
    model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])
    
    for layer in model.layers[:]:
        layer.trainable = True
    
    return model    

pretrained_model = create_functional_model()

pretrained_model.load_weights("dog_breed_classifier_model.h5")

# def upload_and_predict2(filename):
#     img = Image.open(filename)
#     img = img.convert('RGB')
#     img = img.resize((224, 224))
#     print(img.size)
#     # show image
#     plt.figure(figsize=(4, 4))
#     plt.imshow(img)
#     plt.axis('off')
#     # predict
# #     img = imread(filename)
# #     img = preprocess_input(img)
#     probs = pretrained_model.predict(np.expand_dims(img, axis=0))
#     for idx in probs.argsort()[0][::-1][:8]:
#         print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])


# upload_and_predict2(uploaded_file)

def upload_and_predict2(filename):
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    print(img.size)
    # show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    # predict
#     img = imread(filename)
#     img = preprocess_input(img)
    probs = pretrained_model.predict(np.expand_dims(img, axis=0))
    for idx in probs.argsort()[0][::-1][:8]:
        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])

if filename is not None:
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')

    probs = pretrained_model.predict(np.expand_dims(img, axis=0))
    # text = []
    st.image(img, use_column_width=True)
    for idx in probs.argsort()[0][::-1][:8]:
        st.text("{:.2f}%".format(probs[0][idx]*100) +" "+ label_maps_rev[idx].split("-")[-1])

    # st.image(img, use_column_width=True)
    




# import streamlit as st

# import json
# from io import BytesIO

# import numpy as np
# from PIL import Image

# import torch
# from torchvision import transforms
# import pretrainedmodels
# from efficientnet_pytorch import EfficientNet



# with open("imagenet_class_index.json", "r") as f:
#     class_idx = json.load(f)
#     idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# available_models = [
#     "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", 
#     "efficientnet-b5", "efficientnet-b6", "efficientnet-b7", 
#     "alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
#     "densenet121", "densenet169", "densenet201", "densenet161",
#     "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext101_32x4d", "resnext101_64x4d", 
#     "squeezenet1_0", "squeezenet1_1", "nasnetamobile", "nasnetalarge", 
#     "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", 
#     "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d",
#     "inceptionv4", "inceptionresnetv2", "xception", "fbresnet152", "bninception",
#     "cafferesnet101", "pnasnet5large", "polynet"
# ]

# def load_moel(model_name):
#     if "efficientnet" in model_name:
#         model = EfficientNet.from_pretrained(model_name)    
#     else:
#         model = pretrainedmodels.__dict__[model_name](num_classes=1000)
#     return model

# option = st.selectbox(
#     'Select Model',
#      available_models)
# model = load_moel(option)
# model.eval()

# # load data
# uploaded_file = st.file_uploader("Choose a Image")

# if uploaded_file is not None:
#     bytes_data = uploaded_file.getvalue()
#     image = Image.open(BytesIO(bytes_data)).convert("RGB")
#     img_for_plot = np.array(image)
    
#     img = transforms.ToTensor()(image)
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#     img = normalize(img).unsqueeze(dim=0)   
#     result = model(img).squeeze(dim=0)
#     predict_idx = result.argmax().item()
#     prob = torch.softmax(result, dim=0)
#     st.image(img_for_plot, use_column_width=True)
#     st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")
