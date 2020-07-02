# Universal Value Iteration Networks

## paper
https://aaai.org/Papers/AAAI/2020GB/AAAI-ZhangL.10191.pdf

## video
https://youtu.be/VtHzlrRVSS0

## requirements
* python 3.6.7
* pytorch 1.1.0
* numpy 1.16.4
* scikit-learn 0.21.3
* tensorboardX 1.7

## included tasks
* regular maze
* chessboard maze
* mars maze
* deterministic pmst
* stochastic pmst

## train via IL
``` python train_IL.py --config_file config/[task name]_IL.json```

## train via RL
``` python train_RL.py --config_file config/[task name]_RL.json```
