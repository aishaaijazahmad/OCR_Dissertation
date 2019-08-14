from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import os
import os.path
from os import listdir
from PIL import ImageFont
from PIL import Image
from IPython.display import display
from PIL import ImageDraw
import random
import numpy as np
import cv2
import time
import glob
import sys
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
from pathlib import Path
from scipy.misc import toimage
from sklearn.metrics import confusion_matrix
import itertools


#letter_dir = 'base_letters/'
img_size = 32
batch = 64
epoch = 25
wiggle = 5
base_size = 100
min_font = 24
max_font = 32

samples_per_class = 5

#samples_per_class = 32 * 16

input_shape = (img_size, img_size, 1)
font_dir = 'fonts/'
save_dir = 'C:/Users/aisha/Desktop/research/python/letters/'
charset = 'abcdef'
#charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

num_classes = len(charset)


def add_gaussian_noise(image):
    gauss = np.random.normal(loc=0, scale=20, size=image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    temp = noisy > 255
    noisy[temp] = 255
    temp = noisy < 0
    noisy[temp] = 0
    return noisy


def crop_image_random(image):
    x = round(base_size / 2 - img_size / 2)
    y = round(base_size / 2 - img_size / 2)
    y += random.randint(0, wiggle)  # add vertical variance
    return image[y:y + img_size, x:x + img_size]


def add_line(image, color):
    y1 = random.randint(16, 30)
    y2 = y1 + random.randint(-4, 4)
    image = cv2.line(image, (0, y1), (31, y2), color=color, thickness=1)
    return image


def make_random_image(font_file, c, idx):
    paper = random.randint(140, 255)
    ink = random.randint(0, 90)
    font_size = random.randint(min_font, max_font)
    underscore = bool(random.getrandbits(1))
    font = ImageFont.truetype(font_dir + font_file, font_size)
    left = random.choice(charset)
    right = random.choice(charset)
    string = left + c + right
    canvas = Image.new('RGBA', (base_size, base_size), (paper, paper, paper))
    draw = ImageDraw.Draw(canvas)
    w, h = draw.textsize(text=string, font=font)
    w = round((base_size - w) / 2)
    h = round((base_size - h) / 2)
    draw.text((w, h), string, (ink, ink, ink), font=font)
    ImageDraw.Draw(canvas)
    canvas = np.asarray(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    canvas = crop_image_random(canvas)
    canvas = add_gaussian_noise(canvas)

    if underscore:
        canvas = add_line(canvas, ink)
    filename = font_file.lower().replace('.ttf', '') + '_%s_%s_%s.png' % (charset.index(c), c, idx)
    #canvas.save(save_dir + filename)
    cv2.imwrite(save_dir+filename, canvas)
    
    img=mpimg.imread(save_dir+filename)
    #imgplot = plt.imshow(img, cmap = "gray")
    #plt.show()
    return canvas, img


def generate_samples():
    random.seed()
    fonts = os.listdir(font_dir)
    results = []
    images = []
    for font in fonts:
        for character in charset:
            label = charset.index(character)
            print('GENERATING SAMPLES - %s - %s' % (font, character))
            for i in range(samples_per_class):
                sample, img = make_random_image(font, character, i)
                results.append((sample, label))
                '''
                images.append(img)
                imgplot = plt.imshow(img, cmap = "gray")
                plt.show()
                '''
    return results


def instantiate_model():
    print('COMPILING MODEL')
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def prepare_datasets(samples):
    random.shuffle(samples)
    training_set = samples[:round(len(samples) / 10 * 9)]
    test_set = samples[round(len(samples) / 10 * 9):]
    
    ''''-------
    print('look here')
    print(test_set)
    print('ends here')
    #-------'''
    training_data = [i[0] for i in training_set]
    training_data = np.asarray(training_data).astype('float32')
    training_data = np.expand_dims(training_data, axis=3)
    training_data /= 255
    training_labels = [i[1] for i in training_set]
    training_labels = np.asarray(training_labels)
    training_labels = np_utils.to_categorical(training_labels, num_classes)
    print('TRAINING DATA SHAPE', training_data.shape)
    print('TRAINING LABELS SHAPE', training_labels.shape)
    test_data = [i[0] for i in test_set]
    test_data = np.asarray(test_data).astype('float32')
    test_data = np.expand_dims(test_data, axis=3)
    test_data /= 255
    test_labels = [i[1] for i in test_set]
    test_labels = np.asarray(test_labels)
    test_labels = np_utils.to_categorical(test_labels, num_classes)
    print('TEST DATA SHAPE', test_data.shape)
    print('TEST LABELS SHAPE', test_labels.shape)
    return training_data, training_labels, test_data, test_labels

#---------------------------cm from sklearn website-----------------------------------

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



#---------------------------cm from sklearn website-----------------------------------

if __name__ == '__main__':
    
    t0 = time.time()
    samples = generate_samples()
    t1 = time.time()
    
    time_taken = t1-t0
    print('Time taken is ', "%.2f" % time_taken, ' seconds.')
    print('SAMPLES GENERATED', len(samples))
    
    training_data, training_labels, test_data, test_labels = prepare_datasets(samples)
    classifier = instantiate_model()
    
    t2 = time.time()
    history = classifier.fit(training_data, training_labels, validation_split = 0.33, batch_size=batch, epochs=epoch, verbose=1)
    t3 = time.time()
    
    score = classifier.evaluate(test_data, test_labels, batch_size=batch)
    print('OVERALL SCORE', score)
    print("%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))

    time_taken_2 = (t3-t2)/60
    print('Time taken is ', "%.2f" % time_taken_2, ' minutes.')
    
    classifier.save('C:/Users/aisha/Desktop/research/python/keras_alphanumeric.mod')
    

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epoch
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

    
    
    #----------------testing
    splice_test_data = test_data[slice(1,11,1)]
    
    '''----------load-----------this section works but the model is not expecting 32,32,4 rather 32,32,1. Why is this mismatch happening?
    
    filepath = 'C:/Users/aisha/Desktop/brandnew/'
    imglist = listdir(filepath)
    loaded = []
    for image in imglist:
        #ignore desktop.ini file
        if 'ini' in image:
            continue
        else:
            img = Image.open(filepath+image)
            display(img)
            img2 = np.asarray(img)
            loaded.append(img2)
    splice_test_data = loaded
    #--------------'''        
    
    #----------load complete?-----------
    
    prediction = classifier.predict(splice_test_data)
    
    vowels = 'aeioAEIOrRlLhHfFsSXxnNmM8'
    consonants = 'qwtypuUkjgdzcvbQWTYPKJGDZCVB012345679'
    
    
    
    print("Individual Predictions")

    
    
    i=1
    for single_prediction in prediction:
        print('prediction', i)
        print(single_prediction)
        myimage = splice_test_data[i-1]
        #print('the actual image: ', myimage)
        print('Image shape: ', myimage.shape)
        
        
        
        maximum = np.amax(single_prediction)
        maximum = maximum*100
        maxpos = np.argmax(single_prediction)
        
        coi = charset[maxpos]
        if coi in vowels:
            print('This is ', "%.2f" % maximum, '% an', charset[maxpos])
        elif coi in consonants:
            print('This is ', "%.2f" % maximum, '% a', charset[maxpos])

        i = i+1
'''        
      
    print("Confusion Matrix")
    rounded_prediction = classifier.predict_classes(test_data, batch_size = 10, verbose = 0)
    cm = confusion_matrix(test_labels, rounded_prediction)
    cm_plot_labels = charset
    plot_confusion_matrix(cm, cm_plot_labels, title = "Confusion Matrix")
    
  '''
