# About
Codes for paper **Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach.**

*Tiulpin, A., Thevenot, J., Rahtu, E., Lehenkari, P., & Saarakkala, S. (2018). Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach. Scientific reports, 8(1), 1727.*

## Background
Osteoarthritis (OA) is the 11th highest disability factor and it is associated with the cartilage and bone degeneration in the joints. The most common type of OA is the knee OA and it is causing an extremely high economical burden to the society while being difficult to diagnose. In this study we present a novel Deep Learning-based clinically applicable approach to diagnose knee osteoarthritis from plain radiographs (X-ray images) outperforming existing approaches.

## Attention maps examples
Our model learns localized radiological findings as we imposed prior anatomical knowledge to the network architecture.
Here are some examples of attention maps and predictions (ground truth for the provided images is Kellgren-Lawrence grade 2):

<img src="https://github.com/lext/DeepKnee/blob/master/pics/15_2_R_1_1_1_3_1_0_own.jpg" width="260"/> <img src="https://github.com/lext/DeepKnee/blob/master/pics/235_2_R_3_3_0_0_1_1_own.jpg" width="260"/>  <img src="https://github.com/lext/DeepKnee/blob/master/pics/77_2_R_2_0_0_0_0_1_own.jpg" width="260"/>

## This repository includes

- [x] Codes for the main experiments (Supplementary information of the article);
- [x] Pre-trained models;
- [x] Datasets generation scripts;
- [x] MOST and OAI cohorts bounding box annotations;
- [x] Conda environments;
- [x] Support of the inference on the external data.

## Usage
This repository includes the training code and the pre-trained models from each of our experiments. Please, see the paper for more details.

### Setting up the environment
For our experiments we used Ubuntu 14.04, CUDA 8.0 and CuDNN v6.
 
For the convenience, we provide a script to set up a `conda` environment that should be used for training and inference of the models.
Create, configure, and activate it as follow:

```
$ ./create_conda_env.sh
$ conda activate deep_knee
```

### Inference on your data
To run the inference on your DICOM data (assuming you followed the steps above), do the following:

0. Clone the [KneeLocalizer](https://github.com/MIPT-Oulu/KneeLocalizer) repository, and produce 
the file with the bounding boxes, which determine the locations of the knees in the images
(for the detailed instructions, see KneeLocalizer repository README file);
1. Clone the `DeepKnee` repository locally:
    ```
    $ git clone git@github.com:MIPT-Oulu/DeepKnee.git
    $ cd DeepKnee
    ```
2. Fetch the pre-trained models:
    ```
    $ git lfs install && git lfs pull
    ```
3. Create 16bit .png files of the left and right knees from the provided DICOMs:
    ```
    $ cd Dataset 
    $ python crop_rois_your_dataset.py --help
    $ python crop_rois_your_dataset.py {parameters}
    $ cd ..
    ```
    **NOTE:** the image of the left knee will be horizontally flipped to match the right one.
4. Produce the file with KL gradings of the extracted knee images:
    ```
    $ cd own_codes
    $ python inference_own/predict.py --help
    $ python inference_own/predict.py {parameters}
    $ cd ..
    ```

### Training on your data
To run the training, execute the corresponding Bash files from the project directory 
(validation is visualized in `visdom`). 

## License
This code is freely available only for research purposes.

## How to cite
```
@article{tiulpin2018automatic,
  title={Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach},
  author={Tiulpin, Aleksei and Thevenot, J{\'e}r{\^o}me and Rahtu, Esa and Lehenkari, Petri and Saarakkala, Simo},
  journal={Scientific reports},
  volume={8},
  number={1},
  pages={1727},
  year={2018},
  publisher={Nature Publishing Group}
}
```
