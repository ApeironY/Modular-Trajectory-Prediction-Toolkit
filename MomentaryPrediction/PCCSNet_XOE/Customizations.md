# Instructions of how to implement the XOE to PCCSNet #

This document describes in detail how to implement the XOE (both AOE and MOE) to PCCSNet.

## Network Structures ##

The structure of PCCSNet mainly includes 5 parts: Observation (past) Encoder (ObsEncoder), Prediction (future) Encodeer (PredEncoder), Decoder, Classifier and Synthesizer. 

To implement the XOE to PCCSNet, just replace the ObsEncoder with XOE.
