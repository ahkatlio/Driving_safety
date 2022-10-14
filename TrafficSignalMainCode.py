#imports

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import pandas as pd
import random 
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#constants

path = "myData"
labelFile = "labels.csv"
batch_size_val= 50
steps_per_epoch_val = 2000
epochs_val = 10
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

#importingImages

count = 0
images = []
classNo = []
myList = os.listdir(path)
print('Total Classes Detected:', len(myList))
noOfClass = len(myList)
print("Importing Classes.......")
for x in range (0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImage = cv2.imread(path + "/" + str(count) +"/"+y)
        images.append(curImage)
        classNo.append(count)
    print(count, end = " ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

#splitData

X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=testRatio) 
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validationRatio)
steps_per_epoch_val = len(X_train)//batch_size_val
validation_steps = len(X_test)//batch_size_val

#X_train is array of image train
#Y_train is class id

#check image and number of label

print("Data Shapes")
print("Train", end = ""); print(X_train.shape, Y_train.shape)
print("Validation", end = ""); print(X_validation.shape, Y_validation.shape)
print("Test", end = ""); print(X_test.shape, Y_test.shape)
assert(X_train.shape[0] == Y_train.shape[0])
assert(X_validation.shape[0] == Y_validation.shape[0])
assert(X_test.shape[0] == Y_test.shape[0])
assert(X_train.shape[1:] == (imageDimensions))
assert(X_validation.shape[1:] == (imageDimensions))
assert(X_test.shape[1:] == (imageDimensions))

#read csv file

data = pd.read_csv(labelFile)
print("data shape", data.shape, type(data))

#display

num_of_samples = []
cols = 5
num_classes = noOfClass
fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (5, 300) )
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[Y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap = plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "." + row["Name"])
            num_of_samples.append(len(x_selected))

#plot

print(num_of_samples)
plt.figure(figsize = (12,4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

#image processing

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img 
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])

#depth of 1

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

dataGen= ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)
batches= dataGen.flow(X_train,Y_train,batch_size=20)
X_batch,Y_batch = next(batches)
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0],imageDimensions[1]))
    axs[i].axis('off')
plt.show()
 
 
Y_train = to_categorical(Y_train,noOfClass)
Y_validation = to_categorical(Y_validation,noOfClass)
Y_test = to_categorical(Y_test,noOfClass)

#convolution

def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5)
    size_of_Filter2=(3,3)
    size_of_pool=(2,2) 
    no_Of_Nodes = 500
    model= Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(imageDimensions[0],imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClass,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model = myModel()
print(model.summary())
history=model.fit_generator(dataGen.flow(X_train,Y_train,batch_size=batch_size_val),steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,validation_data=(X_validation,Y_validation),shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,Y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])

#pickel
model.save('model_trained.h5')
model = load_model('model_trained.h5')
#pickle_out= open("Traffic Signal/model_trained.p","wb")
#pickle.dump(model,pickle_out)
#pickle_out.close()
cv2.waitKey(0)








