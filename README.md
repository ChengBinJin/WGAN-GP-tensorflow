# WGAN-GP-tensorflow
This repository is a Tensorflow implementation of the [WGAN-GP](https://arxiv.org/abs/1704.00028) for MNIST, CIFAR-10, and ImageNet64.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/46613989-a53ea080-cb4f-11e8-83c1-a8b99dc7bc5b.png" width=800)
</p>  
  
* *All samples in README.md are genearted by neural network except the first image for each row.*

## Install Prerequisites

*   python 3.5, 3.6 or 3.7
*   python3-tk

Ubuntu/Debian/etc.:

    sudo apt install python3.5 python3.5-tk

## Create Virtual Environment

    python -m venv venv

## Activate Virtual Environment

Windows:

    venv/Scripts/activate

Bash:

    source venv/bin/activate

## Install Virtual Environment Requirements

    pip install -r requirements.d/venv.txt

## Create Execution Environments

    tox --notest

That will install tensorflow which uses only the CPU.

To use an Nvidia GPU:

    .tox/py35/bin/python -m pip uninstall tensorflow
    .tox/py35/bin/python -m pip install tensorflow-gpu==1.13.1
    .tox/py36/bin/python -m pip uninstall tensorflow
    .tox/py36/bin/python -m pip install tensorflow-gpu==1.13.1
    .tox/py37/bin/python -m pip uninstall tensorflow
    .tox/py37/bin/python -m pip install tensorflow-gpu==1.13.1

To use an AMD GPU:

    .tox/py35/bin/python -m pip uninstall tensorflow
    .tox/py35/bin/python -m pip install tensorflow-rocm==1.13.1
    .tox/py36/bin/python -m pip uninstall tensorflow
    .tox/py36/bin/python -m pip install tensorflow-rocm==1.13.1
    .tox/py36/bin/python -m pip uninstall tensorflow
    .tox/py37/bin/python -m pip install tensorflow-rocm==1.13.1

## Generated Images
### 1. Toy Dataset
Results from 2-dimensional of the 8 Gaussian Mixture Models, 25 Gaussian Mixture Models, and Swiss Roll data. [Ipython Notebook](https://github.com/ChengBinJin/WGAN-GP-tensorflow/tree/master/src/jupyter).  

**Note:** To demonstrate following experiment, we held the generator distribution Pg fixed at the real distribution plus unit-variance Gaussian noise.
- **Top:** GAN discriminator  
- **Middle:** WGAN critic with weight clipping  
- **Bottom:** WGAN critic with weight penalty  
<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46775237-55bfc680-cd41-11e8-84ee-21d793f56631.gif' width=1000>
  </a>
</p>

**Note:** For the next experiment, we did not fix generator and showed generated points by the generator.
- **Top:** GAN discriminator     
- **Middle:** WGAN critic with weight clipping  
- **Bottom:** WGAN critic with weight penalty  
<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46775252-6a9c5a00-cd41-11e8-8646-8778b9561519.gif' width=1000>
  </a>
</p>

### 2. MNIST Dataset
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/47339737-1dbc6a00-d6d7-11e8-83bc-d499d06912fc.png" width=900>
</p>

<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/47339786-3f1d5600-d6d7-11e8-98ec-bc19c9532168.png" width=900>
</p>

### 3. CIFAR-10
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/47658992-537db900-dbd7-11e8-86d9-ba9ea3273d9e.png" width=900>
</p>

<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/47659033-655f5c00-dbd7-11e8-94e2-363b9b980eff.png" width=900>
</p>

### 4. IMAGENET64
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48756003-cd86f680-ecda-11e8-949f-ee1a8cee8426.png" width=900>
</p>

<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48756029-e4c5e400-ecda-11e8-97c5-49e806ffada6.png" width=900>
</p>

## Documentation
### Download Dataset
'MNIST' and 'CIFAR10' dataset will be downloaded automatically from the code if in a specific folder there are no dataset. 'ImageNet64' dataset can be download from the [Downsampled ImageNet](http://image-net.org/small/download.php).

### Directory Hierarchy
``` 
.
│   WGAN-GP
│   ├── src
│   │   ├── imagenet (folder saved inception network weights that downloaded from the inception_score.py)
│   │   ├── cache.py
│   │   ├── cifar10.py
│   │   ├── dataset.py
│   │   ├── dataset_.py
│   │   ├── download.py
│   │   ├── inception_score.py
│   │   ├── main.py
│   │   ├── plot.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   ├── utils.py
│   │   └── wgan_gp.py
│   Data
│   ├── mnist
│   ├── cifar10
│   └── imagenet64
```  
**src**: source codes of the WGAN-GP

### Training WGAN-GP
Use `main.py` to train a WGAN-GP network. Example usage:

```
python main.py
```
 - `gpu_index`: gpu index, default: `0`  
 - `batch_size`: batch size for one feed forward, default: `64`  
 - `dataset`: dataset name from [mnist, cifar10, imagenet64], default: `mnist`  
 
 - `is_train`: training or inference mode, default: `True`  
 - `learning_rate`: initial learning rate for Adam, default: `0.001`  
 - `num_critic`: the number of iterations of the critic per generator iteration, default: `5`
 - `z_dim`: dimension of z vector, default: `128`
 - `lambda_`: gradient penalty lambda hyperparameter, default: `10.`
 - `beta1`: beta1 momentum term of Adam, default: `0.5`
 - `beta2`: beta2 momentum term of Adam, default: `0.9`

 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `inception_freq`: calculation frequence of the inception score, default: `1000`
 - `sample_batch`: number of sampling images for check generator quality, default: `64`
 - `load_model`: folder of save model that you wish to test, (e.g. 20181120-1558). default: `None` 

### WGAN-GP During Training
**Note:** From the following figures, the Y axises are tge negative critic loss for the WGAN-GP.
1. **MNIST**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/47659728-e5d28c80-dbd8-11e8-96bb-762d9555636c.png" width=900>
</p>

2. **CIFAR10**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/47659471-66dd5400-dbd8-11e8-8f9a-47d42420e816.png" width=900>
</p>

3. **IMAGENET64**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/48756901-518ead80-ecde-11e8-89dc-7e586db34b9a.png" width=900>
</p>

### Inception Score on CIFAR10 During Training  
**Note:** Inception score was calculated every 1000 iterations.
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/47659769-ff73d400-dbd8-11e8-8976-9563d2d50c2d.jpg" width=600>
</p>

### Test WGAN-GP
Use `main.py` to test a WGAN-GP network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20181120-1558
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018wgan-gp,
    author = {Cheng-Bin Jin},
    title = {WGAN-GP-tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/WGAN-GP-tensorflow}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [igul222](https://github.com/igul222/improved_wgan_training).
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer).

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [Vanilla GAN](https://github.com/ChengBinJin/VanillaGAN-TensorFlow)
- [DCGAN](https://github.com/ChengBinJin/DCGAN-TensorFlow)
- [WGAN](https://github.com/ChengBinJin/WGAN-TensorFlow)
