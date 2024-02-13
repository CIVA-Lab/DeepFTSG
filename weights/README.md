# Pre-trained weights of DeepFTSG
This folder contains pre-trained wights of DeepFTSG. If you want to use pre-trained weights, put them inside **Src/weights/** folder.

```DeepFTSG_1.pt``` is an early fusion (appearance + BGS + Flux (motion cues)) single stream trained network model using data from CDnet-2014.

```DeepFTSG_2.pt``` is an early (BGS + Flux (motion cues)) and middle (appearance + motion cues) fusion multiple stream trained network model using data from CDnet-2014.