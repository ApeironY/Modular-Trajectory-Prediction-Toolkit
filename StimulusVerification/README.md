# Stimulus Verifier #

This folder contains the codes for CVPR2023 paper "[Stimulus Verification is a Universal and Effective Sampler in Multi-modal Human Trajectory Prediction]()".

## Environment ##

The codes in this folder are developed with PyTorch and CUDA support. As the codes only use relatively basic functionalities, 
no specific version settings are required.


## Dataset ##

The Stimulus Verifier currently shares the same `dataset` folder as our prior work PCCSNet, while only using the ETH/UCY dataset.

Specifically, in folder `../dataset/ethucy`, you will find:

- folder `processed`: containing pre-processed trajectories stored in `.npy` files, *not used for stimulus verifier*.
- folder `raw`: raw `.csv` files of the ETH/UCY dataset, each containing 4 rows of data (frame, ID, x, y). 
  In addition to the training process, these files are also used for generating social data as a pre-processing step.
  See paragraph below.
- <a id="semantics"> folder `semantic_maps` </a>: manually annotated semantic maps stored in `.npy` format, each containing a matrix M that has the 
  same shape (H, W) as the scene image, where M[i, j] = 0 / 1 indicates whether there is an obstacle (therefore impossible 
  to walk on) at that particular location of the scene image.
- file `ethucy.py` processing codes for PCCSNet, *not used for stimulus verifier*.

## Preparation for Social Data ##

After cloning the project, first run `python preprocess_social_data.py`. This will preprocess social 
data for the Social Verifier and save them in the `social_data/preprocessed` folder.

## Evaluation ##

We provide pre-trained models of both Context Verifier and Social Verifier as well as detailed evaluation 
configurations on the ETH/UCY dataset in [`StimulusVerifier_AdditionalFiles.zip`](https://drive.google.com/file/d/1K2Lp8eX5LpRQqk8Sr0lx_Shookn7FMSp/view?usp=sharing).

After downloading and unzipping it under folder `StimulusVerifier`, you will find:

- `base_model_outputs`: The outputs of base prediction models. In the folder `PCCSNet`, you will find the 
   prediction files of our prior work PCCSNet on the ETH/UCY dataset, each containing 200 predictions in absolute 
   coordinates. Besides, the ground-truth files are also provided. Note that these ground-truth files shares the same 
   content as the `.npy` files under `../dataset/ethucy/processed` (concatenated if necessary).
   Besides, the same predictions can also be acquired by running the 'test' command of PCCSNet using our public models.
- `saved_models`: Trained verifier models, separately placed in folders named `context` and `social`.
- `scores_cache`: Cached likelihoods of PCCSNet's predictions using pre-trained verifiers.
- `verification_configs.py`: Detailed configurations of stimulus verification on PCCSNet's predictions. Note that such 
   configurations can be different for other base prediction models' predictions.

Then, the evaluation can be carried out by running

``python verify_predictions.py -bm PCCSNet -d <DATASET_NAME>``

where `<DATASET_NAME>` is the name of the dataset, e.g. `eth` or `zara1`. See [`datasets.py`](datasets.py) for details.

The evaluation will give the following results (ADE/FDE):

| Dataset | AP Before Verification | AP After Verification | FP Before Verification | FP After Verification |
| :----: | :----: | :----: | :----: | :----: |
| ETH | 0.26 / 0.51 | 0.25 / 0.49 | 0.29 / 0.43 | 0.28 / 0.41 |
| HOTEL | 0.11 / 0.19 | 0.10 / 0.17 | 0.12 / 0.16 | 0.11 / 0.15 |
| UNIV | 0.29 / 0.60 | 0.26 / 0.52 | 0.32 / 0.53 | 0.29 / 0.44 |
| ZARA1 | 0.21 / 0.44 | 0.20 / 0.41 | 0.24 / 0.38 | 0.22 / 0.35 |
| ZARA2 | 0.15 / 0.33 | 0.14 / 0.31 | 0.17 / 0.29 | 0.16 / 0.26 |
| Avg | 0.20 / 0.41 | 0.19 / 0.38 | 0.23 / 0.36 | 0.21 / 0.32 |

Here AP stands for ADE-prioritized, where the trajectory with minimum ADE among all 20 predictions is used for evaluation;
whereas FP stands for FDE-prioritized, which means that the trajectory with minimum FDE among all 20 predictions is used.

## Training ##

To train a stimulus verifier from scratch, use the following command

``python train_stimulus_verifier.py -st <STIMULUS_TYPE> -d <DATASET_NAME>``

where `<STIMULUS_TYPE>` can be `context` or `social` in our current implementation. For other possible hyper-parameters,
please refer to `train_stimulus_verifier.py`. 

## Customizations ##

If you wish to train Stimulus Verifier using your own data, here are some potentially helpful reminders.

- For both types of stimulus verifier, our recommendation is that you place your data under `../dataset/<NEW_DATA>`, and 
  organize the files similar to the file structure of `../dataset/ethucy`. Meanwhile, you should also specify the mappings 
  from dataset names to actual raw data files (see [`dataset_info.py` L1-8](dataset_info.py) for details).
- Social Verifier: For social data, if you followed our recommendation above and kept the raw data files under `../dataset/<NEW_DATA>/raw`
  identical to ETH/UCY `.csv` files, you can prepare the social data using `python preprocess_social_data.py` with some changes
  in arguments. Otherwise [`preprocess_social_data.py`](preprocess_social_data.py) needs to be modified so that it supports your own data format.
- Context Verifier: To train a context verifier, you need to first prepare a semantic map for your data. In our 
  implementation, we use a single-channel 'image' to indicate that (see [`semantic_maps` in Dataset](#semantics)). Yet multi-channel 
  semantic maps are also allowed after modifying the CNN structure of context verifier in `model/models.py` accordingly.
  Besides, a set of transformations that translates annotated coordinates to pixel coordinates in the scene image (if 
  applicable) is needed (see [`dataset_info.py` L10-19](dataset_info.py)).

If you wish to verify predictions with trained verifiers, here are also some reminders.

- Social Verification: Before verifying social stimulus, please make sure that the social information for each of the 
  trajectories to be verified can be generated or has been prepared in advance. In our implementation, the social information
  of base model outputs is already prepared during the execution of `python preprocess_social_data.py`.
- Context Verification: When verifying context stimulus of candidate trajectories, please make sure that the trajectories 
  are in absolute coordinates instead of relative ones so that the coordinate transformations can work properly.
