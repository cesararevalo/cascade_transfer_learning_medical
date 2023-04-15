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



# NOTEBOOK Setup

In order to be able to run jupyter locally for the notebook I had to do the following setup:

```
% python3.10 -m pip install --upgrade pip
% pip install pandas
% pip install "torch>=1.7.0+cpu" "torchvision>=0.8.1+cpu" -f https://download.pytorch.org/whl/torch_stable.html
% pip install ~/Downloads/
% pip install -U "ray[air]"
% pip install opencv-python
% pip install sklearn
% pip install -U scikit-learn
% brew install xz
% pip install -r requirements.txt
```

References:
* https://doc.cocalc.com/howto/custom-jupyter-kernel.html

# MODELS

```bash
Python 3.10.10 (main, Mar  8 2023, 16:54:11) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.path.append('./model/sourcemodel')
>>> sys.path.append('../Cascade_Transfer_Learning/model/sourcemodel')
>>> import torch
>>> import os
>>> import Build_Network
>>> model3 = torch.load(os.path.join(f'/Users/cesar/Documents/university/illinois/cs-598-deep_learning_for_healthcare/project/code/cascade_transfer_learning_medical/model/sourcemodel/SourceNetwork/layer 3/trained model'), map_location='cpu').module
>>> print(model3)
Sequential(
  (0): Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (9): non_first_layer_cascade_Net(
    (new_conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv1_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (pool_aux): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc1): Linear(in_features=12800, out_features=256, bias=True)
    (drop): Dropout2d(p=0.0, inplace=False)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (fc3): Linear(in_features=256, out_features=23, bias=True)
  )
)
>>> 
>>> model4 = torch.load(os.path.join(f'/Users/cesar/Documents/university/illinois/cs-598-deep_learning_for_healthcare/project/code/cascade_transfer_learning_medical/model/sourcemodel/SourceNetwork/layer 4/trained model'), map_location='cpu').module
>>> 
>>> print(model4)
Sequential(
  (0): Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (11): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (12): non_first_layer_cascade_Net(
    (new_conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv1_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (pool_aux): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc1): Linear(in_features=6272, out_features=256, bias=True)
    (drop): Dropout2d(p=0.0, inplace=False)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (fc3): Linear(in_features=256, out_features=23, bias=True)
  )
)
>>> 
>>> model5 = torch.load(os.path.join(f'/Users/cesar/Documents/university/illinois/cs-598-deep_learning_for_healthcare/project/code/cascade_transfer_learning_medical/model/sourcemodel/SourceNetwork/layer 5/trained model'), map_location='cpu').module
>>> 
>>> print(model5)
Sequential(
  (0): Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (11): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
  (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (14): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (15): non_first_layer_cascade_Net(
    (new_conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv1_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (pool_aux): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc1): Linear(in_features=3200, out_features=256, bias=True)
    (drop): Dropout2d(p=0.0, inplace=False)
    (fc2): Linear(in_features=256, out_features=256, bias=True)
    (fc3): Linear(in_features=256, out_features=23, bias=True)
  )
)
>>> 
```

# REFERENCES

* https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start
* 