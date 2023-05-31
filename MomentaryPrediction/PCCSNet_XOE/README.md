# PCCSNet with XOE #

This folder contains the codes for PCCSNet with XOE (both AOE and MOE).

## Basic Setups ##
All of our codes are run on Python 3.7 and PyTorch 1.4.0 with CUDA 10.1 support.

## Results & Pre-trained Models ##

Pre-trained models of PCCSNet with AOE or MOE are available [here](https://drive.google.com/drive/folders/1ZPeJsPtF8kFHQnXMnQePls3fogcmIxEz?usp=share_link), which will
give the following `PCCSNet+AOE Arbitrary` and `PCCSNet+MOE Momentary` results (min ADE / min FDE, k = 20) after evaluation. 

Note that the results are ADE-Prioritized results.

| Dataset | PCCSNet Traditional | PCCSNet Momentary | PCCSNet Arbitrary | PCCSNet+MOE Momentary | PCCSNet+AOE Arbitrary |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ETH | 0.28 / 0.54 | 0.34 / 0.65 | 0.33 / 0.60 | 0.31 / 0.57 | 0.31 / 0.56 |
| HOTEL | 0.11 / 0.19 | 0.14 / 0.25 | 0.20 / 0.36 | 0.13 / 0.21 | 0.22 / 0.41 |
| UNIV | 0.29 / 0.60 | 0.31 / 0.63 | 0.25 / 0.54 | 0.25 / 0.53 | 0.24 / 0.52 |
| ZARA1 | 0.21 / 0.44 | 0.23 / 0.46 | 0.21 / 0.44 | 0.20 / 0.41 | 0.20 / 0.42 |
| ZARA2 | 0.15 / 0.34 | 0.16 / 0.37 | 0.16 / 0.34 | 0.14 / 0.31 | 0.16 / 0.33 |
| ETH/UCY Avg | 0.21 / 0.42 | 0.24 / 0.47 | 0.23 / 0.46 | 0.20 / 0.41 | 0.23 / 0.45 |
| SDD | 8.62 / 16.16 | 9.19 / 17.71 | 8.80 / 16.91 | 8.40 / 16.08 | 7.91 / 15.14 |


## Evaluation ##
To evaluate the pre-trained models, first download the pre-trained models and unzip the files into the `saved_models` folder. 
Then use the following commands:

For AOE:

For ETH/UCY:

`python PCCSNet_MOE/main.py -d <DATASET_IDX> -o 5 -k 20 --flip_aug -aoe -ntc --add_behavior --add_intention --add_empirical`

For SDD:

`python PCCSNet_MOE/main.py -df SDD -d <DATASET_IDX> -o 5 -k 20 --flip_aug -aoe -ntc --add_behavior --add_intention --add_empirical --grid_size 40`

For MOE:

For ETH/UCY:

`python PCCSNet_MOE/main.py -d <DATASET_IDX> -o 5 -k 20 --flip_aug -moe -ntc`

For SDD:

`python PCCSNet_MOE/main.py -df SDD -d <DATASET_IDX> -o 5 -k 20 --flip_aug -moe -ntc --grid_size 40`


where `DATASET_IDX` is the target dataset index (See [notations](#notations) for details). Note that it is possible to use multiple indexes for consecutive 
evaluations, for example:

`python main.py -d 0 2 4 -o 5 -k 20 --flip_aug --rotate -ntc`

## Training from Scratch ##

For AOE:

To train models from scratch, use the command

`python PCCSNet_MOE/main.py -d <DATASET_IDX> -train -k 20 --flip_aug -aoe -pm <PAST FEATURE WEIGHT> --add_behavior --add_intention --add_empirical`

this is equivalent to

`python PCCSNet_MOE/main.py -d <DATASET_IDX> -o 0 1 2 3 4 5 -k 20 --flip_aug -aoe -pm <PAST FEATURE WEIGHT> --add_behavior --add_intention --add_empirical`

You can add `--add_behavior`, `--add_intention`, `--add_empirical` to the command to enable these modules in AOE.

For MOE:

To train models from scratch, use the command

`python PCCSNet_MOE/main.py -d <DATASET_IDX> -train -k 20 --flip_aug -moe -pm <PAST FEATURE WEIGHT>`

this is equivalent to

`python PCCSNet_MOE/main.py -d <DATASET_IDX> -o 0 1 2 3 4 5 -k 20 --flip_aug -moe -pm <PAST FEATURE WEIGHT>`

You can add `-df SDD` to the command if you wish to train on SDD data.
Note that since the XOE output tends to have larger values compared to LSTM outputs, we added the PAST FEATURE WEIGHT `pm`, which is a constant that multiplies to the XOE output during clustering in order to deal with the unbalanced weight problem. For ETH/UCY data, `pm` is around 0.02; 
and for SDD data, `pm` is around 0.0025.

By default, the models will be automatically saved to `saved_models` folder. It is also possible to change it by add `-sd SAVE_DIR`
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

For the operations:

| Operation idx | Operation Name |
| :----: | :----: | 
| 0 | Train Past Encoder |
| 1 | Train Future Encoder |
| 2 | Train Decoder |
| 3 | Train Classifier |
| 4 | Train Synthesizer |
| 5 | Evaluation |
| 6 | Test (Generate Prediction) |
| \> 6 | No Operations (Return) |
