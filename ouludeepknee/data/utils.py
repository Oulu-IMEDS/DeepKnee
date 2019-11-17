import os

import cv2
import numpy as np
import pydicom as dicom


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def dicom_img_spacing(data):
    spacing = None

    for spacing_param in ["Imager Pixel Spacing", "ImagerPixelSpacing", "PixelSpacing", "Pixel Spacing"]:
        if hasattr(data, spacing_param):
            spacing_attr_value = getattr(data, spacing_param)
            if isinstance(spacing_attr_value, str):
                if isfloat(spacing_attr_value):
                    spacing = float(spacing_attr_value)
                else:
                    spacing = float(spacing_attr_value.split()[0])
            elif isinstance(spacing_attr_value, dicom.multival.MultiValue):
                if len(spacing_attr_value) != 2:
                    return None
                spacing = list(map(lambda x: float(x), spacing_attr_value))[0]
            elif isinstance(spacing_attr_value, float):
                spacing = spacing_attr_value
        else:
            continue

        if spacing is not None:
            break
    return spacing


def read_dicom(filename, spacing_none_mode=True):
    """
    Reads a dicom file
    Parameters
    ----------
    filename : str or pydicom.dataset.FileDataset
        Full path to the image
    spacing_none_mode: bool
        Whether to return None if spacing info is not present. When False the output of the function
        will be None only if there are any issues with the image.
    Returns
    -------
    out : tuple
        Image itself as uint16, spacing, and the DICOM metadata
    """

    if isinstance(filename, str):
        try:
            data = dicom.read_file(filename)
        except:
            raise UserWarning('Failed to read the dicom.')
            return None
    elif isinstance(filename, dicom.dataset.FileDataset):
        data = filename
    else:
        raise TypeError('Unknown type of the filename. Mightbe either string or pydicom.dataset.FileDataset.')

    img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)

    if data.PhotometricInterpretation == 'MONOCHROME1':
        img = img.max() - img
    try:
        img = img.reshape((data.Rows, data.Columns))
    except:
        raise UserWarning('Could not reshape the image while reading!')
        return None

    spacing = dicom_img_spacing(data)
    if spacing_none_mode:
        if spacing is not None:
            return img, spacing, data
        else:
            raise UserWarning('Could not read the spacing information!')
            return None

    return img, spacing, data


def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    """
    This function changes the histogram of the image by doing global contrast normalization

    Parameters
    ----------
    img : array_like
        Image
    cut_min : int
         Low percentile to trim
    cut_max : int
        Highest percentile trim
    multiplier : int
        Multiplier to apply after global contrast normalization

    Returns
    -------
    array_like
        Returns a processed image

    """

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier

    return img
