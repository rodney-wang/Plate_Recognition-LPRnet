import cv2
import random
from scipy import ndimage
import numpy as np

def random_padding(image):
    max_pad_w = 10
    max_pad_h = 5

    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, h_pad[0], h_pad[1],  w_pad[0], w_pad[1], cv2.BORDER_CONSTANT,
                                value=color)
    return new_im

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def augment_data(image):
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)
    :param image: Tensor
    :param max_rotation: float, maximum permitted rotation (in radians)
    :return: Tensor
    """

    image = random_padding(image)

    bright = np.random.randint(-120, 120) #typically between [-127, 127]
    contrast = np.random.randint(-60, 60) #typically between [-64, 64]
    apply_brightness_contrast(image, bright, contrast)

    image = ndimage.rotate(image, (np.random.rand()-0.5)*14)   #random rotate -7 to 7 degree
    if random.random()>0.5:
        image = 255-image

    return image



if __name__ == "__main__":

    img = cv2.imread('/Users/fei/tmp/plate.jpg')
    image = augment_data(img)
    cv2.imwrite('/Users/fei/tmp/plate_aug.jpg', image)

