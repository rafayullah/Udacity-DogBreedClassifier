# Dog Breed Identification


## Project Definition
The project demonstrates how transfer learning can be applied to a given problem to improve the accuracy and efficiency of the models. The project utilizes various concepts to prove that with limited time and resources, models can be developed. The models utilized use the Imagenet dataset to train and identify the categories present in the data. We couple the trained weights with our own model as an input to further identify dog breeds.

Additionally, a web application is developed that makes use of trained models. Provided an image as input identifies if a human or dog is present. If Dog is present, it identifies the dog's breed. If a human is present, it identifies what dog breed does the human resemble.

## Files
**dog_app.ipynb** contains a jupyter notebook demonstrating data analysis and model development
**webapplication** folder has a developed simple web application for this project

## Installation
Install required libraries using the requirements file using the following instruction.
```
pip install -r requirements.txt 
```

## Instructions to run the web application
1. go to Webapp's directory
```
cd webapplication
```
2. Run the following command to start webapp
```
python app.py

```
By default, the app runs at http://0.0.0.0:3001/

## Web Application Dashboard
![Main page](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Dashboard_Image.png?raw=true)



## Problem Statement
### Identification of dog breeds
### Description: 
The project targets to deliver an analysis of various approaches to develop models for the classification of various dog breeds present in an input image. Additionally, due to the integration of transfer learning, the model is expected to have efficient and accurate classification using limited resources and training time.
At project will also include a web application for the purpose of demonstration.

## Data Exploration
Dog images to train model can be found at following:
[link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

The dataset contains dog images from 133 different breeds. Train and Validation sets are used for training and fine-tuning the model while for evaluation, a Test set is used.
Below is an overview of the dataset.
* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.

Characteristics (visualized in the next section):
- Images in the datasets have dogs at different angles
- Some breeds have strong resemblance to one another
- Some breeds have different variance in colours

## Data Visualisation 
1. The dataset contains some breeds that very closely resemble each other. Even humans will find it difficult to tell the difference between the two dog classes in some categories. An example is shown below:
- Brittany Breed
![Brittany_breed](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Brittany_02625.jpg?raw=true)
- Welsh Springer Spaniel Breed
![Welsh_springer_spaniel](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Welsh_springer_spaniel_08203.jpg?raw=true)

2. Some of the breeds like Labrador have different variants of colours like yellow, dark brown and black:
![Labrador colours](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Labrador_Colours.png?raw=true)



## Metrics
Since we are dealing with a multi-classification problem here and the data is slightly imbalanced, I used accuracy evaluation metric and categorical_crossentropy cost function. But, first, the labels have to be in a categorical format. The target files are the list of encoded dog labels related to the image with this format. This multi-class log loss punishes the classifier if the predicted probability leads to a different label than the actual and cause higher accuracy. A perfect classifier has a loss of zero and an accuracy of 100%.

## Proposition
* Step 1: Detect Human
* Step 2: Detect Dog
* Step 3: Compare performance of various CNN architectures to classify dog breeds 
    * a: CNN From Scratch
    * b: Transfer learning using VGG16
    * c: Transfer learning using InceptionV3
* Step 4: Final prediction

### Step 1: Detect Human
OpenCV's implementation of:
[Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 

OpenCV provides many pre-trained face detectors, stored as XML files on GitHub. Haarcascades directory contains a pre-trained face detector.
If a human image is provided to the module, it returns the coordinates around the detected face.
So this OpenCV classifier is used only to detect humans and for dog detection, another classifier is used.

### Step 2: Detect Dog
A pre-trained ResNet-50 model is used:
[ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 

The model is trained on [ImageNet](http://www.image-net.org/) containing over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction for the object that is contained in the image.
This model was able to detect dogs in images with an accuracy of 100%.

We could have used the above-mentioned ResNet based model for detecting the human presence but the ImageNet dataset does not explicitly cover humans as a category.

### Step 3: Compare performance of different CNNs architectures to classify Dog Breeds 
Now that we have identified that if a dog or human is present in the image or not, the next step is to identify its breed. CNN's are used for this purpose.
Following are the experimentations performed:
a. CNN From Scratch
A CNN model is built from scratch with the following architecture.
A model having 4 convolutional layers following maxpool layers was trained for 20 epochs. 
The model achieved an accuracy of above 10%.
b. Transfer learning using VGG16
To reduce training time without sacrificing accuracy, transfer learning is used. VGG16 is used as a pre-trained model to obtain bottleneck features of each input image. A new model is created and trained to consist of GlobalAveragePooling layer and the final Dense layer to predict 133 dog breeds. 
The model achieved an accuracy of above 40%.
c. Transfer learning using InceptionV3
InceptionV3 is also used as a pre-trained model to calculate bottleneck features which are then fed to a second model consisting of GlobalAveragePooling layer following two fully connected dense layers with dropout to predict 133 dog breeds.
The model achieved an accuracy of above 80%.

### Step 4: Final prediction
The final function that is used to predict dog breeds works in the following way:
1. Detects if the provided image is either a human, dog or has an absence of both of these
2. If a dog is detected, a predicted breed is returned
3. If a human is detected, a predicted breed is returned resembling that human 
4. If human and dog are not detected, a relevant message is returned

## Model Evaluation and Validation
I have used the pretrained InceptionV3 model on ImageNet for this task.
- To transfer its learning to my model, I have used the trained weigts as an input to a GAP layer which reduces the image size to half, keeping the prominent features only
- Before adding the final output layer, I have added an intermediary fully connected layer of 1024 units to improve the performance further. A dropout layer is also added to avoid any overfitting

![Model results](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Final_Arch.png?raw=true)

- A dropout of 0.4 is selected to avoid model overfitting sacrificing model's performance on test set
- Model is finally compiled with categorical_crossentropy loss function due to being a multi-class classification
- Model is trained for 20 epocs
- The model achieves an accuracy of above 80% on 836 test dog images


## Justification
To the images provided as an input to our model, we were returned with the following classifications:
![Model results](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Model_Results.png?raw=true)

- In the first car image, the model is clearly able to distinguish the absense of human or a dog present
- In the second image, the model identifies the image to be a human and returns a dog breed that it resembles:
![Icelandic sheepdog](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Icelandic_sheepdog.jpg?raw=true)
- In the third image, provided an image of a Rottweiler, the model confuses it with Beauceron:
![Beauceron](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Beauceron.jpg?raw=true)

We can clearly see the resemblance between the two breeds
- In the fourth image, the model identifies the cat to be a strongly resembling dog's breed Affenpinscher
- In the last image, the model correctly identifies an English cocker spaniel

Considering the random guess has a probablity of <1% to be correct, with and accuracy of 84% our model has performed really well.

## Reflection
At the start of the project our objective was to create a CNN that can achieve a performance of upto 80%.
As we started our model development we realized that this could not be achieved by a model that is not very deep. And certainly training a model that has such qualities will take much more resources in terms of training time and effort. This came out to be challenge on how we can achieve a performance of at least 80% while keeping the training times low.
However, we came out with a solution to integrate transfer learning in our pipeline. With transfer learning, we could simply use a pre-trained model having a adequately deep architecture. We finally choose InceptionV3 based model trained on Imagenet dataset. This means that we can simply download this model from the web and simply load its pre trained as an input to our pipeline.
The results were very interesting, this let us achive a test accuracy in the range of 84%, and our model was quickly able to not only distinguish between the different breeds but also provided an input image of a human, it could find the resemblance with a dog.
With the experiment, it became apparent that how transfer learning can be applied to increase the models performance.

## Improvements
The output is better than I expected. The transfer learning kicks in when we try to predict human resembling a dog, where we can see how the model is capable of comparing certain human features like hair and skin with the provided dog breeds.
Improvements:
    -When provided a cat, the dog detection model predicts it to be a dog, albeit a dog resembling closely to the provided cat image. This means that the model can see a bit improvement with objects resmbling dogs.
    -Human detector works great but when provided a complex image, it breaks down and does not detect a human presense
    -Dog breed classifier works properly, however it confuses with multiple similar breeds. To resolve this, we can output multiple higher probablility classes rather than single having highest probability in such cases.
    -Model can be further made efficient by using Early stopping based on val_loss
    -Data augmentation can be added to further enhance the datasets since dogs in pictures tend to have many variations in angle

## Conclusion
* Transfer learning is very useful for the datasets having similarities among them
* It greatly reduces the time and resources required to train a complex model
* Considering the random guess has a probablity of <1% to be correct, with and accuracy of 84% our model has performed really well.



## Authors
* [Rafay Ullah Choudhary](https://github.com/rafayullah)
