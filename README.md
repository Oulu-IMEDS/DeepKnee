# About
Codes for paper **Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach.**

*Aleksei Tiulpin, Jérôme Thevenot, Esa Rahtu, Petri Lehenkari, and Simo Saarakkala*, 2017.

## Background

Osteoarthritis (OA) is the 11th highest disability factor and it is associated with the cartilage and bone degeneration in the joints. The most common type of is knee OA and it is causing an extremly high economical burden to the society while being difficult to diagnose. In this study we present a novel Deep Learning-based approach to diagnose knee osteoarthritis from plain radiographs (X-ray images). 

## Benchmarks and how-to-run

Here we present the training codes and the pretrained models from each of our experiments. Please, see the paper for more details.

To train the networks, you should install the following main dependencies:

* Python 3
* pytorch with CUDA support
* PIL
* matplotlib
* Jupyter Notebook (to work with attention maps)
* tqdm
* visdom
* numpy
* termcolor
* torchvision

We recommend to create a virtual environment with the aforemention packages. To run the training, run the corresponding bash files (validation is visualized in visdom). Before running, edit the beginnign of the file to activate your virtual environment.

However, you can run the codes as they are and just use the parameters fixed in the bash scripts.

## Attention maps examples
Our model learns localized radiological findings as we imposed prior anatomical knowledge to teh network architecture. Here are some examples of attention maps and predictions (Kellgren-Lawrence grade 2 ground truth):

<img src="https://github.com/lext/DeepKnee/blob/master/pics/15_2_R_1_1_1_3_1_0_own.jpg" width="260"/> <img src="https://github.com/lext/DeepKnee/blob/master/pics/235_2_R_3_3_0_0_1_1_own.jpg" width="260"/>  <img src="https://github.com/lext/DeepKnee/blob/master/pics/77_2_R_2_0_0_0_0_1_own.jpg" width="260"/> 
## License

This code is freely available only for research purpuses. 

