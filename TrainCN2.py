from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,MaxPool2D
# from numpy.testing import assert_allclose
# import os

num_classes = 3
img_rows,img_cols = 224,224
batch_size = 64

train_data_dir = 'DATA_TRAIN_2_224_224/train'
validation_data_dir = 'DATA_TRAIN_2_224_224/val'

train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='rgb',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='rgb',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)





model = Sequential()

# VGG 16

model.add(Conv2D(input_shape=(img_rows,img_cols,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

model.summary()

from keras.optimizers import RMSprop,SGD,Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)





# from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



'''earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)'''

nb_train_samples =    6000	#4500		#7566				#5532
nb_validation_samples = 1200 #900	#1947			#2341
epochs= 6


# #1
# # define the checkpoint
# filepath = "TH7.h5"
# checkpoint = ModelCheckpoint(filepath,
#                              monitor='val_loss',
#                              mode='min',
#                              save_best_only=True,
#                              verbose=1)
# callbacks_list = [checkpoint]
# # fit the model
# model.fit(
#                 train_generator,
#                 steps_per_epoch=nb_train_samples//batch_size,
#                 epochs=epochs,
#                 callbacks=callbacks_list,
#                 validation_data=validation_generator,
#                 validation_steps=nb_validation_samples//batch_size)

#2
# load the model
filepath = "TH7.h5"

new_model = load_model(filepath)

#assert_allclose(model.predict(train_generator),new_model.predict(train_generator), 1e-5)

# fit the model
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)
callbacks_list = [checkpoint]
new_model.fit(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

