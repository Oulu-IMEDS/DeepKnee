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
