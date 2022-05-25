# AlphaZero - Reinforcement Learning

Simple Reinforcement Learning with AlphaZero, based on DeepMind's AlphaGo Zero<sup>**[1]**</sup>.

## Source files

### AlphaZero implementation

* `Arena.py`: Compare the newly trained model with the best one.
* `MCTS.py`: Monte Carlo Tree Search implementation.
* `NNet.py`: AlphaZero neural network architecture.
* `SelfPlay.py`: Generate neural network input features playing games with currently the best agent to be used in the unsupervised learning process.
* `Train.py`: Reinforcement learning training procedure.

### Supervised learning

* `dataset_builder.py`: Generate neural network input features from a dataset for supervised learning.
* `train_alphazero.py`: Train an initial model using the input features resulting from the dataset. 

### Open-source

* `features.py` & `go.py`: Open-source go board game implementation _as is_ from Brian Lee & Andrew Jackson (https://github.com/tensorflow/minigo).
* `Gnugo.py`: Connection with the Go Text Protocol of GNU Go implementation from Laurent Simon.

## Requirements

This project requires dependencies according to your willingness to use or GPU or not. To take the most profit out of this project, please consider using a GPU, the latter greatly speeding up the NNet-related features.

### CPU-only

For a CPU-only use, the following environnement should do the trick.
* Python 3.8
* tensorflow (2.6.0)
* tqdm

### GPU

#### NVIDIA CUDA Toolkit

Dependencies for an NVIDIA GPU use greatly varies according to your GPU. However, you'll need to have CUDA Toolkit; Please refer to _NVIDIA CUDA Toolkit Documentation_ for installation:
* Windows : https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
* Linux : https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

#### NVIDIA cuDNN

Having installed CUDA, the next step is to download and install _cuDNN_. Downloading can be made at the following link:
* https://developer.nvidia.com/cudnn

Unzip cuDNN files and copy them to their respective CUDA folders.

At this point, you should check that the NVIDIA toolkit has been added to your PATH. You may also need to reboot your computer.

#### Setting up the python environnement

Now that CUDA and cuDNN are installed, you should be allset for a proper python environnement installation. A minimalist environnement for this project is:
* Python 3.8
* cudatoolkit (version depends on your GPU)
* cudnn (version depends on your CUDA version)
* tensorflow-gpu (2.6.0)
* tqdm

A sample procedure to create a conda environnement could be (for a GTX 1660 Super/CUDA 11.3):
```shell
conda create --name tf2 python=3.8.0
conda activate tf2
conda search cudatoolkit
conda search cudnn
conda install cudatoolkit=11.3.1 cudnn=8.2.1 -c=conda-forge
pip install --upgrade tensorflow-gpu==2.6.0
conda install -c conda-forge tqdm
pip install keras==2.6
```

Note that `pip install keras==2.6` may or may not be useful, yet, the keras version should correspond to the tensorflow's one.

#### Testing the environnement

You can test your installation with the following python code:
```python
import tensorflow as tf
print(tf.config.list_physical_devices())
```
The GPU should appear in the list.


## Usage

### Supervised learning

A supervised learning process can be initiated, using GNUGo's scores to set policy's values, and win probability for the value. The process is divided in two stages:

**1.** Generate the dataset.
```shell
python dataset_builder.py
```

**2.** Train the model.
```shell
python train_alphazero.py
```

This procedure will generate an initial model `trained.h5` to start with, located under `model/`.

_Nota Bene_: This process can take a while to complete if using a CPU-only environnement.

### Unsupervised learning

The unsupervised learning process is the most important one in this project. It is the process of training a neural network to play against itself. The process is divided in three steps:
* Self-play: Generate neural network input features from a dataset.
* Network training: Train an initial model using the input features resulting from the dataset.
* Comparison: Compare the newly trained model with the best one.

If you want to use the unsupervised learning process, you can run the following command:
```shell
python Train.py
```
Hyper-parameters can be changed in the `Config.py` file. This process should iteratively strengthen the `best.h5` model.

_Nota Bene_: This process can take a while to complete if using a CPU-only environnement.

### Comparison to the random agent

The best model can be compared to the random agent, using the following command:
```shell
python CompareToRandom.py
```

## Implementation details

## Results

## References

* **[1]** Silver, David & Schrittwieser, Julian & Simonyan, Karen & Antonoglou, Ioannis & Huang, Aja & Guez, Arthur & Hubert, Thomas & Baker, Lucas & Lai, Matthew & Bolton, Adrian & Chen, Yutian & Lillicrap, Timothy & Hui, Fan & Sifre, Laurent & Driessche, George & Graepel, Thore & Hassabis, Demis. (2017). Mastering the game of Go without human knowledge. Nature. 550. 354-359. 10.1038/nature24270.
* **[2]** Wang, Hui & Emmerich, Michael & Preuss, Mike & Plaat, Aske. (2019). Hyper-Parameter Sweep on AlphaZero General. 