# Customizing the PCCSNet #

This document describes in detail the components of the PCCSNet and therefore serves as a guide for user customizations.


## Data Formats ##
Our implementation of the PCCSNet takes trajectories of shape (`num_trj`, `trj_len`, 2) as inputs, in which `num_trj` is the total number
of trajectories, `trj_len` is the length of each trajectory (obs_len + pred_len), 
and the last dimension indicates the location of the agent using 
x,y-axis coordinates.

Note that the function [`get_data`](utils/data_process.py#L7) will also return a list (`split_marks`), this is used to calculate the ground truths for 
classifier using modality loss as it indicates which parts of the train set are 
extracted from the same actual scene.


## Network Structures ##

The structure of PCCSNet mainly includes 5 parts: Observation (past) Encoder (ObsEncoder), Prediction (future) Encoder (PredEncoder), 
Decoder, Classifier and Synthesizer. We now describe each of them in detail.

### Observation & Prediction Encoder ###
Both ObsEncoder and PredEncoder take trajectories as inputs and output the corresponding deep features.

In our implementation, we use LSTMs as the ObsEncoder as well as the PredEncoder 
(see class [`EncoderLSTM`](models/components.py#L6) for details). To train the encoders, 
we assign a corresponding decoder (also LSTM in our case, abandoned afterwards) to each of these encoders 
and wrap them in a the [`EncoderTrainer`](models/auxiliary.py#L9) class.

### Decoder ###
The Decoder takes concatenated deep features of both the observation and the prediction and outputs the predicted trajectory. 

Same as the encoders, we use LSTM to implement the Decoder (see class [`DecoderLSTM`](models/components.py#L22) for details.)
Note that [`DecoderLSTM`](models/components.py#L22) is also the model used as the corresponding decoders for the encoders above. 


### Classifier ###

Our classifier is implemented as a simple MLP which takes the observed encodings as input and outputs 
a soft-maxed probability distribution among all the modalities. See class [`Classifier`](models/components.py#L38) for details.

### Synthesizer ###

The synthesizer in our codes consists of 2 fully-connected layers, it is used to fuse the data from both the past representation and 
the modality representation together into the future representation. See class [`Synthezier`](models/components.py#L81) for details.
