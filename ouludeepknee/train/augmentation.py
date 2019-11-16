"""
Data augmentations

(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import random
from PIL import Image, ImageEnhance
import numbers
import numpy as np


class CenterCrop(object):
    """
    Performs center crop of an image of a certain size.
    Modified version from torchvision
    
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        tw, th,  = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


def correct_gamma16(img, gamma):
    """
    Gamma correction of a 16-bit image
    """
    img = np.array(img).astype(np.float64)
    img = (img/65535.)**gamma
    img = np.uint16(img*65535)
    img = Image.fromarray(img)
    return img


def correct_gamma8(img, gamma):
    """
    Gamma correction of an 8-bit image
    """
    img = np.array(img).astype(np.float64)
    img = (img/255.)**gamma
    img = np.uint8(img*255)
    img = Image.fromarray(img)
    return img


class CorrectGamma(object):
    """
    Does random gamma correction
    
    """
    def __init__(self, g_min, g_max, res=8):
        self.g_min = g_min
        self.g_max = g_max
        self.res = res

    def __call__(self, img):
        gamma = random.random()*self.g_max+self.g_min
        if self.res == 8:
            return correct_gamma8(img, gamma)
        return correct_gamma16(img, gamma)
    
    
class Jitter(object):
    """
    Makes a crop of a fixed size with random offset
    
    """
    def __init__(self, crop_size, j_min, j_max):
        self.crop_size = crop_size
        self.j_min = j_min
        self.j_max = j_max
        
    def __call__(self, img):
        x1 = random.randint(self.j_min, self.j_max)
        y1 = random.randint(self.j_min, self.j_max)
        return img.crop([x1, y1, x1+self.crop_size, y1+self.crop_size])

    
class Rotate(object):
    """
    Performs random rotation
    
    """
    def __init__(self, a_min, a_max, interp=Image.BICUBIC):
        self.a_min = a_min
        self.a_max = a_max
        self.interp = interp
        
    def __call__(self, img):
        angle = random.uniform(self.a_min, self.a_max)
        return img.rotate(angle,resample=self.interp)


class CorrectBrightness(object):
    """
    Performs random brightness change
    
    """
    def __init__(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max
        
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(self.b_min, self.b_max)
        return enhancer.enhance(factor)


class CorrectContrast(object):
    """
    Performs random contrast change
    
    """
    def __init__(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max
        
    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(self.b_min, self.b_max)
        return enhancer.enhance(factor)
