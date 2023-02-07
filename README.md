# Deep Cross-Modal Image-Text Matching

Hi! This is the code repository of the project for **ECE 570 (Fall 2021)** on **Deep Cross-Modal Image-Text Matching**.

### Repository Structure:
```
src
├── agents
│   ├── base.py
│   ├── joint_feature_adv.py
│   ├── joint_feature_cp.py
│   ├── joint_feature.py
│   ├── joint_features_2.py
│   └── joint_features_bert.py
├── configs
│   └── exp_resnet101_bilstm_764emb_pretrained_64_Adam2e-4_w0_min2-50_cp.json
├── data_preprocess
│   ├── data_cub.sh
│   ├── data_cuhk.sh
│   ├── data_flicker30k.sh
│   ├── data_flowers.sh
│   ├── directory.py
│   ├── preprocess_bert.py
│   └── preprocess.py
├── datasets
│   ├── cub_bert.py
│   ├── cub.py
│   ├── cuhk_pedes_2.py
│   ├── cuhk_pedes_bert.py
│   ├── cuhk_pedes.py
│   ├── flickr30k_bert.py
│   ├── flickr30k.py
│   ├── flowers_bert.py
│   └── flowers.py
├── graphs
│   ├── losses
│   │   ├── bce.py
│   │   ├── cross_entropy.py
│   │   ├── huber_loss.py
│   │   └── __init__.py
│   ├── models
│   │   ├── bilstm.py
│   │   ├── discriminator.py
│   │   ├── model.py
│   │   ├── resnet101.py
│   │   └── resnet50.py
│   └── weights_initializer.py
├── main.py
├── README.md
└── utils
    ├── config.py
    ├── dirs.py
    ├── env_utils.py
    ├── env_utils.pyc
    ├── __init__.py
    ├── metrics.py
    ├── misc.py
    └── train_utils.py

```
Inside the "Datasets" folder all the pre-processed images and the text file are stored. However, we haven't included them due to its large size. We provide one of the pre-processed datasets [here](https://purdue0-my.sharepoint.com/:u:/g/personal/pate1332_purdue_edu/EeWz-zMbN51MsumuiAAMQMwBx8lLp7EvKyeeiMO6EKstnQ?e=4hPItd) . The dataset is a **.tar** file which is required to be extracted in the Datasets folder.

## Instructions to run the code.
Open the terminal inside the code repository and run the following commands
```
python3 -m venv <ENV_NAME>
source <ENV_NAME>/bin/activate
pip3 -r requirements.txt
cd src
python3 main.py configs/<EXPERIMENT JSON FILE>
```

## How was the code adopted?
* The official code repository [1] is in Tensorflow, we use that as our primary reference for the project.
* The code repository at [2] is a PyTorch template that we used as a reference to prepare the code-base for the project.
* Lately, at [3] we discover an alternate PyTorch version of the original paper. We have used some of the loss functions and metrics implementation from that. We have additional codes for the BERT encoder and adversarial training which is written on our own. Also, [3] has an implementation bug that computes the recall metrics incorrectly, which we have corrected in our version. 


>[1] https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning
> [2] https://github.com/moemen95/Pytorch-Project-Template
> [3] https://github.com/labyrinth7x/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching



