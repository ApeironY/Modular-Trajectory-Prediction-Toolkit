# PCCSNet #

This folder contains the codes for ICCV2021 paper "**[Three Steps to Multimodal Trajectory Prediction: Modality Clustering, Classification and Synthesis](https://arxiv.org/abs/2103.07854)**"

## Basic Setups ##
All of our codes are run on Python 3.7 and PyTorch 1.4.0 with CUDA 10.1 support.

## Datasets ##
We provide pre-processed datasets as `.npy` files in the `dataset/DATAET_NAME/processed` folder, where `DATASET_NAME` is 
either `ethucy` or `SDD` (Stanford Drone Dataset). Note that the SDD data are originally from the [PECNet](https://github.com/HarshayuGirase/PECNet/tree/master/social_pool_data)
Repository.


## Results & Pre-trained Models ##
Pre-trained models are available [here](https://drive.google.com/file/d/13_8CufufnQKOBBUqr4sUMrdadpUPUQHw/view?usp=sharing), which will
give the following results (min ADE / min FDE, k = 20) after evaluation. 

| Dataset | ADE-Prioritized Results | FDE-Prioritized Results | Equal-Focus Results |
| :----: | :----: | :---: | :----: |
| ETH | 0.26 / 0.51 | 0.29 / 0.43 | 0.26 / 0.43 |
| HOTEL | 0.11 / 0.19 | 0.12 / 0.16 | 0.11 / 0.16 |
| UNIV | 0.29 / 0.60 | 0.32 / 0.53 | 0.29 / 0.53 |
| ZARA1 | 0.21 / 0.44 | 0.24 / 0.38 | 0.21 / 0.38 |
| ZARA2 | 0.15 / 0.33 | 0.17 / 0.29 | 0.15 / 0.29 |
| ETH/UCY Avg | 0.20 / 0.41 | 0.23 / 0.36 | 0.20 / 0.36 |
| SDD | 8.62 / 16.16 | 9.41 / 14.01 | 8.62 / 14.01 |

Typically, there are three different ways of defining the best match(es): the ADE-prioritized best match, the FDE-prioritized best match, 
and also the Equal-Focus best matches. 

'ADE-prioritized' (default) means that the trajectory (among all k predictions) with the minimum ADE is considered as the best match, 
whereas 'FDE-prioritized' takes the trajectory with minimum FDE as the best match.
For the Equal-Focus approach, 
the best matches are selected separately to minimize the ADE and FDE respectively.

After obtaining the best match trajectories, 
we calculate their ADEs and FDEs and consider them as our results. 


It is obvious that ADE-prioritized results tends to have a low ADE, and vice versa for the FDE, whereas
the Equal-Focus Approach can minimize both the ADE and the FDE. Ways of 
changing the mode of evaluation can be found in [notations](#notations).



## Evaluation ##
To evaluate the pre-trained models, first download the pre-trained models and unzip the files into the `saved_models` folder. 
then use the following commands:

For ETH/UCY:

`python main.py -d <DATASET_IDX> -o 5 -k 20 --flip_aug --rotate -ntc`

For SDD:

`python main.py -df SDD -d <DATASET_IDX> -o 5 -k 20 --flip_aug --rotate -c 400 --encoder_layer 1 -ntc`


where `DATASET_IDX` is the target dataset index (See [notations](#notations) for details). Note that it is possible to use multiple indexes for consecutive 
evaluations, for example:

`python main.py -d 0 2 4 -o 5 -k 20 --flip_aug --rotate -ntc`

## Testing ##

To generate and save the predictions of our model for testing, use the command

`python main.py -d <DATASET_IDX> -test -tf <TEST_FILE_NAME> -k 20 --rotate -ntc`

in which `<TEST_FILE_NAME>` is a `.npy` file containing the observed trajectories. Ideally, it should 
have the shape of (`num_trj`, `obs_len`, 2). We placed an embedded `process_file()` function in `data_process.py` to 
help the users matching the data format.


## Training from Scratch ##

To train models from scratch, use the command

`python main.py -d <DATASET_IDX> -train -k 20 --flip_aug --rotate`

this is equivalent to

`python main.py -d <DATASET_IDX> -o 0 1 2 3 4 5 -k 20 --flip_aug --rotate`

You can add `-df SDD` to the command if you wish to train on SDD data.

By default, the models will be automatically saved to `saved_models` folder. It is also possible to change it by add `-sd SAVE_DIR`
to the command, in which case it will be automatically created if non-existent.

Generally, the average training time of the PCCSNet on an RTX 2080Ti GPU would be around 3 hours. However, if the settings of modality loss 
is changed, the running-time will be significantly longer when running for the first time (under the new setting). 
This is due to the fact that each scene in the training data needs to be re-processed to get the new
ground truths for the modality loss. Taking zara1 as an example, the average time consumption for each step is:

| Operation | Time |
| :----: | :----: | 
| Train Past Encoder | ~60 min |
| Train Future Encoder | ~60 min |
| Train Decoder | ~40 min |
| Train Classifier | ~5 min |
| Train Synthesizer | ~10 min |
| Process Scene for Modality Loss | ~120 min |

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

For modes of evaluation:

| Eval Mode | Mode Name |
| :----: | :----: | 
| 0 | ADE-prioritized |
| 1 | FDE-prioritized |
| 2 | Equal-Focus |


## Customizations ##
We also provide a separate file to help the users perform customizations, see [customizations](Customizations.md) for details.


## Citing ##
If you find this repository useful, please cite
```
@InProceedings{Sun_2021_ICCV,
    author    = {Sun, Jianhua and Li, Yuxuan and Fang, Hao-Shu and Lu, Cewu},
    title     = {Three Steps to Multimodal Trajectory Prediction: Modality Clustering, Classification and Synthesis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13250-13259}
}
```
