# Self-Batch Mixup Vision Transformer

Note: This repository was forked and modified from
[google-research/vision_transformer](https://github.com/google-research/vision_transformer).

## Introduction

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.



## Installation

Make sure you have `Python>=3.6` installed on your machine.

For installing [Jax](https://github.com/google/jax), follow the instructions
provided in the corresponding repository linked here. Note that installation
instructions for GPU differs slightly from the instructions for CPU.

Then, install python dependencies by running:
```
pip install -r vit_jax/requirements.txt
```


## How to run code
Please upload Self-Batch Mixup Visual Transformer onto Google Colab.
Please terminate current session before you run each time.

You can tune hyper parameter of mixup rate in the "Setup mixup rate" section. If you wish to tune the hyper parameters of visual transformer model, 
you can tune in the "Fine tune" section.

If you want to change data set between CIFAR-10 and CIFAR-100, you can just change the name in the "Load Dataset" section.







