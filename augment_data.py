import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


scale_jitter = 0.7
max_rotate_angle= 7.
gaussian_blur_range = 1.
rank_blur_range = 3 # rank filter window size
motion_blur_range = 3
pydown_blur_range = 0.7
brightness_jitter = 0.1

def __scale_jitter(img):
    w, h = img.size[0], img.size[1]
    h_scale = np.random.uniform(0.8, 1)
    w_scale = np.random.uniform(0.5, 1)
    nw = int(w * w_scale)
    nh = int(h * h_scale)
    img = img.resize((nw, nh), Image.BICUBIC)

    dx = int(np.random.uniform(0, w - nw))
    dy = int(np.random.uniform(0, h - nh))
    new_img = Image.new('L', (w, h), np.random.randint(0, 255))
    new_img.paste(img, (dx, dy))

    return new_img


def __random_pydown_blur(img):
    w, h = img.size[0], img.size[1]
    h_scale = np.random.uniform(pydown_blur_range, 1)
    w_scale = np.random.uniform(pydown_blur_range, 1)
    nw = int(w * w_scale)
    nh = int(h * h_scale)
    img = img.resize((nw, nh), Image.BICUBIC)
    img = img.resize((w, h), Image.BICUBIC)
    return img


def __random_motion_blur(img, motion_blur_range):
    img_array = np.array(img)
    angle = np.random.randint(0, 360)
    motion_blur_range = np.random.randint(2, motion_blur_range)
    M = cv2.getRotationMatrix2D((motion_blur_range / 2, motion_blur_range / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(motion_blur_range))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (motion_blur_range, motion_blur_range))

    motion_blur_kernel = motion_blur_kernel / motion_blur_range
    img_array = cv2.filter2D(img_array, -1, motion_blur_kernel)

    cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX)
    img = Image.fromarray(img_array)
    return img


def __random_gaussian_blur(img):
    blur = np.random.uniform(1, gaussian_blur_range)
    return img.filter(ImageFilter.GaussianBlur(blur))


def __random_rank_blur(img):
    w, h = img.size[0], img.size[1]
    img = img.resize((2 * w, 2 * h))
    rank = np.random.randint(0, rank_blur_range * rank_blur_range)
    img = img.filter(ImageFilter.RankFilter(rank_blur_range, rank))
    img = img.resize((w, h))
    return img


def __random_brightness(img):
    brightness_jitter_need = np.random.uniform(-brightness_jitter, brightness_jitter)
    img_brightness = ImageEnhance.Brightness(img)
    return img_brightness.enhance(1 + brightness_jitter_need)


def __random_rotate(img, max_rotate_angle):
    w, h = img.size[0], img.size[1]
    max_rotate_angle = max_rotate_angle
    angle_need = np.random.uniform(-max_rotate_angle, max_rotate_angle)
    img = img.rotate(angle_need, expand=1)
    img = img.resize((w, h), Image.BICUBIC)
    return img


def __random_blur(img):
    k = np.random.randint(0, 4)
    if k == 0:
        img = __random_gaussian_blur(img)
    elif k == 1:
        img = __random_rank_blur(img)
    elif k == 2:
        img = __random_pydown_blur(img)
    else:
        img = __random_motion_blur(img, 3)
    return img


def __random_inverse_color(img):
    img_array = np.array(img)
    offset = np.random.randint(-10, 10)
    img_reverse_array = 255 + offset - img_array
    img = Image.fromarray(img_reverse_array)
    return img


def augment_data(img):
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)
    :param image: Tensor
    :param max_rotation: float, maximum permitted rotation (in radians)
    :return: Tensor
    """

    if np.random.random() < 0.3:
        img = __random_blur(img)
    if np.random.random() < 0.3:
        img = __random_brightness(img)
    if np.random.random() < 0.3:
        img = __scale_jitter(img)
    if np.random.random() < 0.3:
        img = __random_rotate(img, max_rotate_angle)
    if np.random.random() < 0.5:
        img = __random_inverse_color(img)

    return img



if __name__ == "__main__":

    img = cv2.imread('/Users/fei/tmp/plate.jpg')
    for i in range(10):
        image = augment_data(img)
        cv2.imwrite('/Users/fei/tmp/aug/plate_aug'+str(i)+'.jpg', image)

