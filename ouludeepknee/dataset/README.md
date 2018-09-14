# Dataset pre-rpocessing functionality

## Getting the detections
We recommend to get the bounding boxes using our repository KneeLocalizer

## Creating the dataset for automatic grading from your own DICOM files
When you have created a list of detections using KneeLocalizer, then you can use `crop_rois_your_dataset.py` to cut 140x140mm ROIs for further processing.