
#1

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python import _pywrap_py_exception_registry




# img_rows,img_cols = 224,224  #Vgg 16  -  TH7.h5
img_rows,img_cols = 227,227  #Alexnet  - TH5.h5
# img_rows,img_cols = 48,48  #Unknown  - Uk.h5
batch_size = 32


nb_test_samples = 1225	#Vgg 16 -  Alexnet  - Unknown

# test_data_dir = 'DATA_TRAIN_2_224_224/test'  # VGG 16
test_data_dir = 'DATA_TRAIN_1/test'		#Alexnet
# test_data_dir = 'DATA_48_48/test'		#Unknown


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
							test_data_dir,
							target_size=(img_rows,img_cols),
							color_mode="rgb",
							batch_size=batch_size,
							shuffle=True,
							class_mode="categorical")

filepath = "TH5.h5"
new_model = load_model(filepath)
Y_pred = new_model.predict(test_generator, nb_test_samples//batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['dislike','like','neutral']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))




#2

# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras import metrics
#
# img_rows,img_cols = 227,227
# batch_size = 32
# nb_validation_samples = 1225
# validation_data_dir = 'DATA_TRAIN_1/test'
# validation_datagen = ImageDataGenerator(rescale=1./255)
# validation_generator = validation_datagen.flow_from_directory(
# 							validation_data_dir,
# 							target_size=(img_rows,img_cols),
# 							color_mode="rgb",
# 							batch_size=batch_size,
# 							shuffle=True,
# 							class_mode="categorical")
#
# filepath = 'TH4a.h5'
# new_model = load_model(filepath)
# score = new_model.evaluate(validation_datagen, nb_validation_samples//batch_size, verbose=0)
# print('Test loss: %.4f'% score[0])
# print('Test accuracy %.4f'% score[1])