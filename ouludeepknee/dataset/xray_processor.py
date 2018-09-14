import os

import pydicom as dicom
import numpy as np
import cv2


def read_dicom(filename):
    """
    This function tries to read the dicom file
    
    Parameters
    ----------
    filename : str
        Path to the DICOM file

    Returns
    -------
    tuple or None
        Returns a tuple, which as an image and a pixel spacing. 
        None is returned if the function was not able to read the file / extract spacing.
    """

    try:
        data = dicom.read_file(filename)
        img = np.frombuffer(data.PixelData,dtype=np.uint16).copy().astype(np.float64)
        if data.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max()-img
        img = img.reshape((data.Rows,data.Columns))
    except:
        return None
    try:
        return img, data.ImagerPixelSpacing[0]
    except:
        pass
    try:
        return img, data.PixelSpacing[0]
    except:
        return None


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


def process_file_or_image(fname=None, dataset_dir=None,
                          image=None, spacing=None,
                          save_dir=None, bbox=None, gradeL=None, gradeR=None,
                          sizemm=140, pad=300, save_vis=True):
    """
    Processes one knee xray and saves left and right images into 16bit png files.
    
    Parameters
    ----------
    fname : str
        Filename of the dicom image.
    dataset_dir : str
        DICOM image root folder.
    image : ndarray

    spacing : float

    save_dir : str
        Where to save the files.
    bbox : array_like
        BBox annotations for left and right joint in the format x1l, y1l, x2l, y2l, x1r, y1r, x2r, y2r.
    gradeL : int
        KL grade for the left joint.
    gradeR : int
        KL grade for the right joint.
    sizemm : float
        Size of the ROI in mm.
    pad : int
        Padding for the xray image. It is very useful in the case when the knee is too close to the edges.
        
    Returns
    -------
    bool
        Returns bool if failure. Saves left and/or right knee joints into given folders.
        
    """
    if (fname is not None) and (dataset_dir is not None):
        res_read = read_dicom(os.path.join(dataset_dir, fname))
        if res_read is None:
            print(f"Read DICOM failed on {fname} {bbox}")
            return True

        image, spacing = res_read
        spacing = float(spacing)
    elif (image is not None) and (spacing is not None):
        pass
    else:
        raise ValueError('Provide arguments with either file or ndarray source')

    leftok = (gradeL >= 0)
    rightok = (gradeR >= 0)
    if (not rightok) and (not leftok):
        print(f"Unsupported KL values, failed on {fname} {bbox}")
        return True

    image = process_xray(image, 5, 99, 65535).astype(np.float)
    sizepx = int(np.round(sizemm / spacing))

    image_padded = np.zeros((image.shape[0] + 2*pad,
                             image.shape[1] + 2*pad))
    image_padded[pad:pad+image.shape[0], pad:pad+image.shape[1]] = image
    
    # This can be refactored
    if leftok:
        x1, y1, x2, y2 = [c + pad for c in bbox[:4]]

        cx = x1+(x2-x1) // 2
        cy = y1+(y2-y1) // 2

        x1 = cx - sizepx//2
        x2 = cx + sizepx//2
        y1 = cy - sizepx//2
        y2 = cy + sizepx//2

        patch = cv2.resize(image_padded[y1:y2, x1:x2], (350, 350),
                           interpolation=cv2.INTER_CUBIC)
        patch -= patch.min()
        patch /= patch.max()
        patch *= 65535
        patch = np.round(patch)
        patch_left = patch.astype(np.uint16)

        if save_vis:
            os.makedirs(os.path.join(save_dir, str(gradeL)), exist_ok=True)
            if fname is not None:
                tmp_fname = os.path.join(save_dir, str(gradeL),
                                         f"{fname.split('.')[0]}_L.png")
            else:
                tmp_fname = os.path.join(save_dir, str(gradeL), "crop_L.png")
            cv2.imwrite(tmp_fname, np.fliplr(patch_left))
    else:
        patch_left = None

    if rightok:
        x1, y1, x2, y2 = [c + pad for c in bbox[4:]]

        cx = x1 + (x2-x1) // 2
        cy = y1 + (y2-y1) // 2

        x1 = cx - sizepx // 2
        x2 = cx + sizepx // 2
        y1 = cy - sizepx // 2
        y2 = cy + sizepx // 2

        patch = cv2.resize(image_padded[y1:y2, x1:x2], (350, 350),
                           interpolation=cv2.INTER_CUBIC)
        patch -= patch.min()
        patch /= patch.max()
        patch *= 65535
        patch = np.round(patch)
        patch_right = patch.astype(np.uint16)

        if save_vis:
            os.makedirs(os.path.join(save_dir, str(gradeR)), exist_ok=True)
            if fname is not None:
                tmp_fname = os.path.join(save_dir, str(gradeR),
                                         f"{fname.split('.')[0]}_R.png")
            else:
                tmp_fname = os.path.join(save_dir, str(gradeR), "crop_R.png")
            cv2.imwrite(tmp_fname, patch_right)
    else:
        patch_right = None

    return patch_left, patch_right
