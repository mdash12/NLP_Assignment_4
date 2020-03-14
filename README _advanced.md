This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Overview

The model is run on 2 layers of Bi-GRU with max-pooling on the output of the Bi-GRU layers.

# Data

Data set is already provided in the task.

# Code Overview


## Changes

In train_advanced.py the config is the same as the basic model config code.

In data.py, the shortest_path is used to get the shortest dependency path.

In the advanced model in model.py, the init method is passed the same arguments as the basic model.


# Usage

## Training
python train_advanced.py --embed-file data/glove.6B.100d.txt --embed-dim 100 --batch-size 10 --epochs 7

## Testing
python predict.py --prediction-file my_predictions.txt --batch-size 10
