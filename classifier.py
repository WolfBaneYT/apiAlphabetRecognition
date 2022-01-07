import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
#fetchopenml to get name of dataset - mnist_784 version 1 and assigning values to X and y
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
#Random state can be any number but is needed for initialising
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=9,train_size=3500,test_size=500)
#Scaling the data
X_trainScale = X_train/255.0
X_testScale = X_test/255.0
#Applying logistic regression data by fitting

#Solver is applied to optimise the problem (Small Data Set = LibLinear (Multiclass has to be over)) (larger data set = saga,sag (Multiclass needs to be multinomial)) multi_class can have auto,over or multinomial and default val is auto (if over, binary val )(auto for default)(in multinomial loss is minimised)
clf = LogisticRegression(solver="saga",multi_class='multinomial').fit(X_trainScale,y_train)
#Function that takes image as prediction and makes prediction by converting to scalar quantity and grey so that it doesnt affect prediction
def get_Prediction(image):
    #Bring image
    im_pil =  Image.open(image)
    #used to convert image into bw (two types - P  or PA if alpha and if greyscale L or LA) palette upto 256 and can convert to rgb and L is greyscale and if  image needs rgb .convertRGB
    image_bw = im_pil.convert('L')
    #Resizing image 28/27
    image_bw_resized = image_bw.resize((22,30),Image.ANTIALIAS)
    #Value assigned is 20
    pixel_filter = 20
    #Getting percentile
    minPixel = np.percentile(image_bw_resized,pixel_filter)
    #Assign number to each image
    image_bw_resized_inverted_scale = np.clip(image_bw_resized-minPixel,0,255)
    #making array of max pixel
    maxPixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scale = np.asarray(image_bw_resized_inverted_scale)/maxPixel
    #creating test sample
    test_sample =  np.array(image_bw_resized_inverted_scale).reshape(1,784)
    #test prediction
    test_pred = clf.predict(test_sample)
    return test_pred[0]
