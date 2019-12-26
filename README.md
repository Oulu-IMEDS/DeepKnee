# About
Codes for paper **Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach.**

*Tiulpin, A., Thevenot, J., Rahtu, E., Lehenkari, P., & Saarakkala, S. (2018). Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach. Scientific reports, 8(1), 1727.*

# Disclaimer

**This branch is only for inference purposes. Re-training is possible only in the master branch!!!**

### Running the software
This code requires the fresh-most docker and docker compose installed.

Execute `sh deploy.sh cpu` to deploy the app on CPU. If you have installed nvidia-docker,
you can also deploy on GPU. The inference is 3 times faster on GPU. To deploy on GPU, run `sh deploy.sh gpu`.

Be careful, this app carries all the dependencies and weighs around 10GB in total.

# Technical documentation

The software is currently composed of four separate loosely-coupled services. Specifically, those are:

1. `KNEEL` - Knee joint and landmark localization (https://arxiv.org/abs/1907.12237). REST microservice, port 5000.
2. `DeepKnee` - Automatic KL grading (this work, https://www.nature.com/articles/s41598-018-20132-7). REST microservice running on port 5001.
3. `Backend broker` - a NodeJS microservice implementing asynchronous communication between microservices and UI (socket.io). It runs on 5002 port.
4. `UI` - User Interface implemented in ReactJS. This part runs on 5003.

The platform is designed so that it is possible to use `KNEEL` and `DeepKnee` separately. Both microservices expect
a `JSON` with `{dicom: <I64>}`, where `<I64>` is the dicom file encoded in `base64`. If you make a request to either of the services,
it needs to be done to `/kneel/predict/bilateral` or `/deepknee/predict/bilateral` for `KNEEL` and `DeepKnee`, respectively.

An example script that uses the platform can be found in the file `analyze_folder.py`.

## License
This code is freely available only for research purposes. Commercial use is not allowed by any means.
The provided software is not cleared for diagnostic purposes.

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
