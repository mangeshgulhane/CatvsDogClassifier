#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 02:10:51 2019

@author: Mangesh Gulhane
"""

import tensorflow as tf;
print(tf.__version__)

import os,shutil
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


original_dataset_dir  = '/home/rashmi/Desktop/Machine_learning/Dataset/Catsvsdogs/dogs-vs-cats/train'
base_dir = '/home/rashmi/Desktop/Machine_learning/Dataset/Catsvsdogs/traindataset'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

train_cat_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cat_dir)


train_dog_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dog_dir)

validation_cat_dir = os.path.join(validation_dir,'cats')
os.mkdir(validation_cat_dir)

validation_dog_dir = os.path.join(validation_dir,'dogs')
os.mkdir(validation_dog_dir)

test_cat_dir = os.path.join(test_dir,'cats')
os.mkdir(test_cat_dir)

test_dog_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_dog_dir)

#Train Data 

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cat_dir,fname)
    shutil.copyfile(src,dst)
    
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_dog_dir,fname)
    shutil.copyfile(src,dst)
    
    
#Validation data

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cat_dir,fname)
    shutil.copyfile(src,dst)
    
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_dog_dir,fname)
    shutil.copyfile(src,dst)    
    
#test data
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_cat_dir,fname)
    shutil.copyfile(src,dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dog_dir,fname)
    shutil.copyfile(src,dst)    
    
 
print('total training cat images:',len(os.listdir(train_cat_dir)))
print('total training dog images:',len(os.listdir(train_dog_dir)))
print('total validation cat images:',len(os.listdir(validation_cat_dir)))
print('total validation dog images:',len(os.listdir(validation_dog_dir)))
print('total test cat images:',len(os.listdir(test_cat_dir)))
print('total test dog images:',len(os.listdir(test_dog_dir))) 


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode = 'binary'
        )

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size = 20,
        class_mode='binary')

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc']
              )
    
history=model.fit_generator(train_generator,steps_per_epoch=100,
                            epochs=30,validation_data=validation_generator,
                            validation_steps=50
                            )


model.save('cats_and_dogs_tmp.h5')


#ploting curves of loss and accuracy during training

acc = history.history['acc']
print(type(acc))
#30 epochs so size of list is 30
print(len(acc))

val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
