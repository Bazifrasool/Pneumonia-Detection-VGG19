# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:59:48 2020

@author: Bazif
"""


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
vgg19_model= keras.applications.vgg16.VGG19()
vgg19_model.summary()
model = Sequential()
for layer in vgg19_model.layers:
    model.add(layer)
model.summary()
model.layers.pop()
model.summary()
for layer in model.layers:
    layer.trainable = False
model.add(Dense(2,activation = "softmax"))
model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics=["accuracy"])
classifier=model
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory(
        './train',
        target_size=(224, 224),
        batch_size=8)

test_set = test_datagen.flow_from_directory(
        './test',
        target_size=(224, 224),
        batch_size=8)

classifier.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=10,
        validation_data=test_set,
        validation_steps=200)
