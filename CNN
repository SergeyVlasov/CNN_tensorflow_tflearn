
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn

TRAIN_DIR = 'F:/img_data/train_geom'
TEST_DIR = 'F:/img_data/test_geom'
IMG_Size = 30
LR=0.001
KERNEL_conv = 8
STRIDES = 4



MODEL_NAME = 'geom-{}-{}.model'.format(LR, '2conv_basic')

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'circle': return [1,0,0]
    elif word_label == 'square': return [0,1,0]
    elif word_label == 'triangle': return [0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_Size, IMG_Size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data) 
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img =cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_Size, IMG_Size))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data
        
train_data = create_train_data()



from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


convnet = input_data(shape=[None, IMG_Size, IMG_Size, 1], name='input')


convnet = conv_2d(convnet, 8, 5, strides=STRIDES, activation='relu')   # activation='LeakyReLU'
convnet = max_pool_2d(convnet, STRIDES)

#convnet = conv_2d(convnet, 16, 5, strides=STRIDES, activation='relu')
#convnet = max_pool_2d(convnet, STRIDES)

#convnet = conv_2d(convnet, 32, 5, strides=STRIDES, activation='relu')
#convnet = max_pool_2d(convnet, STRIDES)

convnet = fully_connected(convnet, 128, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data
test = train_data

X = np.array([i[0] for i in train]).reshape(-1, IMG_Size, IMG_Size, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_Size, IMG_Size, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=50, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)


import matplotlib.pyplot as plt

test_data = process_test_data()

fig = plt.figure()



for num, data in enumerate(test_data):
    # circle [1,0,0]
    # square [0,1,0]
    # triangle [0,0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    
    orig = img_data
    data = img_data.reshape(IMG_Size, IMG_Size,1)
    
    model_out = model.predict([data])[0]

    #print(model_out)
    
  

    if np.argmax(model_out)==0: str_label = 'circle'
    elif np.argmax(model_out)==1:str_label = 'square'   
    elif np.argmax(model_out)==2: str_label = 'triangle'
        
    y.imshow(orig, cmap='gray')    
    plt.title(str_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()    
