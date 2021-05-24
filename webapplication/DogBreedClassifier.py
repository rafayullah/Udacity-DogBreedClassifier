import numpy as np
import cv2                
import matplotlib.pyplot as plt 
from tqdm import tqdm

# Dog Detection Libraries
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing import image                  

# Loading dog breeds 
dog_names = np.load('./models/dog_names.npz')
dog_names = dog_names['arr_0']

                       
                             

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



# Converting image path to a tensor
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)



# Loading ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 



# Loading Imagenet trained Inception model 
InceptionV3_model = load_model('./models/InceptionV3_custom_trained.h5')

from models.extract_bottleneck_features import extract_InceptionV3

def predict_dog_breed(imgPath):    
    bottleneck_feature = extract_InceptionV3(path_to_tensor(imgPath))
    prediction = np.argmax(InceptionV3_model.predict(bottleneck_feature))
    prediction = dog_names[prediction]
    return prediction



def get_prediction(imgPath):
    dog = dog_detector(imgPath)
    human = face_detector(imgPath)
    dog_breed = predict_dog_breed(imgPath)
    
    if dog:
        return 'Dog resembling {}'.format(dog_breed)
    elif human:
        return 'Human resembling {}'.format(dog_breed)
    else:
        return 'Neither a human nor a dog'

