import random
import os

import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2

font_dir = 'fonts/'
save_dir = 'C:/Users/aisha/Desktop/research/python/english_selfgen_db/'
charset = 'abcdefghijklmnop'
samples_per_class = 1

def add_gaussian_noise(image):
    gauss = np.random.normal(loc=0, scale=20, size=(32,32,4))
    #gauss = gauss.resize(image.size)
    noisy = image + gauss
    
    #temp is a boolean value
    #true if noisy > 255 else false
    temp = noisy > 255
    noisy[temp] = 255
    
    temp = noisy < 0
    noisy[temp] = 0
    
    # I THINK THIS ABOVE CODE IS A SNAPPING SNIPPET
    # SNAPS THE VALUES TO EITHER 255 OR 0
    
    return noisy


def make_random_image(font_file, c, idx):
    paper = random.randint(150,225)
    ink = random.randint(0,75)
    font_size = random.randint(24,32)
    
    font = ImageFont.truetype(font_dir + font_file, font_size)
    
    string = random.choice(charset) + c + random.choice(charset)
    
    canvas = Image.new('RGBA', (32,32), (paper,paper,paper))
    
    draw = ImageDraw.Draw(canvas)
    
    #the below syntax is known as 'unpacking'
    #THE MORE YOU KNOW BRO
    
    w,h = draw.textsize(text = string, font = font)
    
    #i THINK the above line is returning textsize[0] = width of the text
    #and textsize[1] = height of the text
    
    #now whatever we get, we will minus it from the actual size of the image
    #divide it by 2
    #why? you may ask? to place the text in the center of the image
    #THAT'S WHY
    
    w = round((32 - w)/2)
    h = round((32 - h)/2)
    
    #i don't know what ink is yet. Is it color?
    #YEP IT'S COLOR
    draw.text((w,h),string,(ink,ink,ink), font = font)
    ImageDraw.Draw(canvas)
    
    #IMAGE LAYERING 
    
    #STEP 1 : GAUSSIAN NOISE ADDITION TO IMAGE
    canvas = add_gaussian_noise(canvas)
    
    canvas = Image.fromarray(np.uint8(canvas))
    
    filename = font_file.lower().replace('.ttf', '') + '_%s_%s_%s.png' % (charset.index(c), c, idx)
    
    print(filename)
    canvas.save(save_dir + filename)
    
    
if __name__ == '__main__':
    random.SystemRandom()
    random.seed()
    fonts = os.listdir(font_dir)
    for font in fonts:
        for character in charset:
            for i in range(samples_per_class):
                make_random_image(font, character, i)
