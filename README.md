# Cascade Transfer Learning

Github repository for paper "Deep Cascade Learning for Optimal Medical Image Feature Representation".

## Installation
Please check out requirements.txt and install package accordingly.

MACOS specific additional notes for installation:

* Optional uninstall your python version from pyenv (eg. 3.10.x) (see )
```bash
pyenv uninstall 3.10.x # change x if you have something else
```

* Install xz because it's needed for pandas, and you need this before installing a new python version: 
```bash
brew install xz
```

* Install python version on pyenv:
```bash
pyenv install 3.10
```

* Now you can install pandas:
```bash
pip install pandas
```

* Install some Mac specific requirements if you don't have a GPU:
```bash
pip install "torch>=1.7.0+cpu" "torchvision>=0.8.1+cpu" -f https://download.pytorch.org/whl/torch_stable.html

```

## Reproducing Figure 1(b)
check TCL_IDC.ipynb reproduce the result for Figure 1(b) in the paper. 
If want to train the network, please follow the step below:

## Steps to train the network

First, download dataset in [Here](https://www.kaggle.com/paultimothymooney/breast-histopathology-images).

Second, Download trained models in [Here](https://drive.google.com/drive/folders/1yqCOjaommJvcErzz01LiJaQbX8V6wy2b?usp=sharing)
and put them into SourceNetwork folder.

Third, run:
```bash
```
python TCL_IDC.py --root_dir='./Breast Histopathology Images' --network_address=./model/sourcemodel/SourceNetwork
## License
[MIT](https://choosealicense.com/licenses/mit/)
