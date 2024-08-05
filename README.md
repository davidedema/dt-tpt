![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
<p align='center'>
    <h1 align="center">DT-TPT</h1>
    <p align="center">
    Project for Deep Learning at the University of Trento A.Y.2023/2024
    </p>
    <p align='center'>
    Developed by:<br>
    De Martini Davide <br>
    Rigon Mattia <br>
    Segala Marina <br>
    </p>   
</p>

----------

- [Project Description](#project-description)
- [Installation](#installation)
- [Running the project](#running-the-project)


## Project Description
The main goal was to build a novel method for Test Time Adaptation (TTA). With TTA we refer to techninques that aims to improve the network performance one sample at a time. We choose to develop our TTA algorithm on top of [TPT](https://arxiv.org/pdf/2209.07511) (Test-time Prompt Tuning).

As TPT, the network used for this project is [CLIP](https://arxiv.org/pdf/2103.00020) (Contrastive Learining Image Pre-training) with [CoOp](https://arxiv.org/pdf/2109.01134) (Context Optimization) that is used for improving the performances of models like CLIP (vision language), keeping all the parameters fixed and optimizing the prompt.

Our work focuses on the two part that we think are crucial for TPT:
1.   Sample selection: we developed a method that select the samples in an adaptive way based on the batch entropy.
In the TPT code, there is a fixed confidence selection of samples: they select each time the 10% most secure samples from the batch based on their entropy.
We decided to make this decision in a more dynamic way: the number of selections can vary from 10% to an upper bound calculated based on the local min/max present in the derivative of the batch entropy curve.
It always depends on the entropy values: during the selection, the number of augmentations that are selected are the first N augmentations that minimized the loss.

2. Augmentation of images: for it we propose a method for getting better augmentation using the attention taken from [DINO](https://arxiv.org/pdf/2104.14294) in order to have a guess on which part of the image contains the information.

    Our idea was to take advantage of DINO, seeing that it is a self-supervised learning method that does not require labels.
    It helps us to calculate the attention map of the image selected, before the application of different augmentation on it.
    Indeed, the attention map will underline the main focus of the image and in this way different augmentations can be computed on the basis of that information.

    More specifically we will return a list of images composed by
    - the original image
    - cropped image around its focal point, that has a 30% of probabilities to be horizontally or vertically flipped
    - 'basic' augmented images (the one applied also in the original implementation)
    - list of cropped image with different threashold for the attention

A better explanation could be found inside the notebook.
## Installation

In order to run the project you'll need to clone it and install the requirements. We suggest you to create a virtual environment 
- Clone it

    ```BASH
    git clone https://github.com/dttpt/

    ```
- Create the virtual environment where you want, activate it and install the dependencies 
  
    ```BASH
    cd path/of/the/project
    python -m venv /name/of/virtual/env
    source name/bin/activate
    pip install -r requirements.txt
    ```

## Running the project

The project could be runned in two different ways:
- Through notebook

- Running directly:
  
    ```
    python main.py
    ```
    You can set to use DINO or not in the `flags.py` file under the flag `DINO`.
    The same could be done for the use of "our selection".
