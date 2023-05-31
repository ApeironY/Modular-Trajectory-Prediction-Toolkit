# Momentary Prediction and Arbitrary Prediction #

This folder contains codes of XOE (both AOE and MOE) and implementation of XOE on PCCSNet. 

Refer to [`XOE`](XOE) for codes of XOE and how to pre-train the XOE.

Refer to [`PCCSNet_XOE`](PCCSNet_XOE) for how to implement XOE on PCCSNet and how to train the full model.

Refer to [`DataPreprocess`](DataPreprocess) for the data preprocessing code of ETH/UCY and SDD. You can download the processed data [`here`](https://drive.google.com/drive/folders/1nzjQ7yx_iAjmp1-xrykN3BaC8pM5K51t?usp=share_link) and place them in the "processed_xoe" folder of each dataset in [`../dataset`](../dataset/). 

To preprocess the data of ETH/UCY and SDD on your own, use the following command:

`python DataPreprocess/processor.py`

The processed data will be automatically placed in the "processed_xoe" folder of each dataset in [`../dataset`](../dataset/).
