# XOE #

This folder contains the codes for both AOE and MOE.

## Basic Setups ##

All of our codes are run on Python 3.7 and PyTorch 1.4.0 with CUDA 10.1 support.

## Soft Pre-training ##

To pre-train the XOE on two sub-tasks, you can use the following commands.

For AOE:

For ETH/UCY:

`python XOE/main.py -d <DATASET_IDX> -o 0 1 2 -k 20 --flip_aug -aoe --add_behavior --add_intention --add_empirical`

For SDD:

`python XOE/main.py -df SDD -d <DATASET_IDX> -o 0 1 2 -k 20 --flip_aug -aoe --add_behavior --add_intention --add_empirical`

For MOE:

For ETH/UCY:

`python XOE/main.py -d <DATASET_IDX> -k 20 --flip_aug`

For SDD:

`python XOE/main.py -df SDD -d <DATASET_IDX> -k 20 --flip_aug`

where `DATASET_IDX` is the target dataset index (See [notations](#notations) for details). Note that it is possible to use multiple indexes for consecutive 
evaluations, for example:

`python XOE/main.py -d 0 2 4 -k 20 --flip_aug`

Note that `-o` is only useful for AOE. For detailed operations please refer to [notations](#notations).

By default, the models will be automatically saved to `pretrained_models` folder. It is also possible to change it by add `-sd SAVE_DIR`
to the command, in which case it will be automatically created if non-existent.

## <a id="notations"> Notations </a> ##
For simplicity, we adopted some notations for the operations mentioned above and in the paper.

For ETH/UCY dataset:

| Dataset idx | Dataset Name |
| :----: | :----: | 
| 0 | eth |
| 1 | hotel |
| 2 | univ |
| 3 | zara1 |
| 4 | zara2 |

For SDD dataset:

| Dataset idx | Dataset Name |
| :----: | :----: | 
| 0 | SDD |

For the operations of AOE:

| Operation idx | Operation Name |
| :----: | :----: | 
| 0 | Train AOE Behavior Extractor |
| 1 | Train AOE Intention Extractor |
| 2 | Train full AOE |
| \> 2 | No Operations (Return) |
