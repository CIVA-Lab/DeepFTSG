# DeepFTSG
The official implementation of the IJCV 2023 journal paper [**DeepFTSG: Multi-stream Asymmetric USE-Net Trellis Encoders with Shared Decoder Feature Fusion Architecture for Video Motion Segmentation**](https://link.springer.com/article/10.1007/s11263-023-01910-x)

</br>

## News

**[February 13, 2024]** 

- :fire::fire::fire:  **The src code and weights are uploaded. Instructions on how to train and inference will be added soon.** 

</br>

## DeepFTSG: Multi-Stream Asymmetric USE-Net Trellis Encoders With Shared Decoder Feature Fusion Architecture for Video Motion Segmentation
Discriminating salient moving objects against complex, cluttered backgrounds, with occlusions and challenging environmental conditions like weather and illumination, is essential for stateful scene perception in autonomous systems. We propose a novel deep architecture, named DeepFTSG, for robust moving object detection that incorporates single and multi-stream multi-channel USE-Net trellis asymmetric encoders extending U-Net with squeeze and excitation (SE) blocks and a single shared decoder network for fusing multiple motion and appearance cues. DeepFTSG is a deep learning based approach that builds upon our previous hand-engineered flux tensor split Gaussian (FTSG) change detection video analysis algorithm which won the CDNet CVPR Change Detection Workshop challenge competition. DeepFTSG generalizes much better than top-performing motion detection deep networks, such as the scene-dependent ensemble-based FgSegNet v2, while using an order of magnitude fewer weights. Short-term motion and longer-term change cues are estimated using general-purpose unsupervised methods – flux tensor and multi-modal background subtraction, respectively. DeepFTSG was evaluated using the CDnet-2014 change detection challenge dataset, the largest change detection video sequence benchmark with 12.3 billion labeled pixels, and had an overall F-measure of 97%. We also evaluated the cross-dataset generalization capability of DeepFTSG trained solely on CDnet-2014 short video segments and then evaluated on unseen SBI- 2015, LASIESTA and LaSOT benchmark videos. On the unseen SBI-2015 dataset, DeepFTSG had an F-measure accuracy of 87%, more than 30% higher compared to the top-performing deep network FgSegNet v2 and outperforms the recently proposed KimHa method by 17%. On the unseen LASIESTA, DeepFTSG had an F-measure of 88% and outperformed the best recent deep learning method BSUV-Net2.0 by 3%. On the unseen LaSOT with axis-aligned bounding box ground-truth, network segmentation masks were converted to bounding boxes for evaluation, DeepFTSG had an F-Measure of 55%, outperforming KimHa method by 14% and FgSegNet v2 by almost 1.5%. When a customized single DeepFTSG model is trained in a scene-dependent manner for comparison with state-of-the-art approaches, then DeepFTSG performs significantly better, reaching an F-Measure of 97% on SBI-2015 (+10%) and 99% on LASIESTA (+11%).


## DeepFTSG-1: Single-stream Early Fusion for Spatiotemporal Change Detection
The proposed DeepFTSG-1 uses motion cues as input computed from multi-modal change detection and flux motion through our fast tensorbased motion estimation  and an adaptive split-gaussian multi-modal background subtraction model respectively. DeepFTSG-1 incorporates a three-channel input processing stream, with the first (red) channel being the appearance (the three-channel RGB color input is converted to grayscale). The motion and change cues corresponding to the current frame computed using a background model based on past frames for the case of slower temporal change and a temporal sliding window of frames for the case of flux motion are assigned to the second (G) and third (B) channels.

![](/figures/DeepFTSG-1-Arch.png)


## DeepFTSG-2: Multi-stream Middle Spatiotemporal Fusion
The proposed DeepFTSG-2 extends DeepFTSG-1 by decoupling appearance-based information from spatiotemporal using multiple input streams. The first input stream receives three-channel RGB color input from the current frame, and the second input stream receives motion and change cues corresponding to the current frame, which is computed using a temporal sliding window of frames for the case of motion and using a background model computed from past frames for the case of change. The two input streams go through two parallel feature extraction modules. The first processing stream (appearance encoder) extracts spatial appearance features using the SEResNet- 50 backbone, and the second processing stream (motion encoder) extracts spatiotemporal, motion, and change-based features using the ResNet-18 backbone. The feature maps generated by these two encoders are then fused and processed through the network’s decoder. The motion and change cues are stacked channel-wise, where the red channel (R) corresponds to the background subtraction mask, the green channel (G) corresponds to the motion mask, and the blue channel (B) is set to 0.

![](/figures/DeepFTSG-2-Arch.png)


## Qualitative Results
The qualitative results of proposed DeepFTSG-1 and DeepFTSG-2 are shown below.

![](/figures/qualitativeResults.PNG)


## Generalization and Extension 
The proposed DeepFTSG network uses a generalized multi-stream architecture that can be readily extended to support additional multimodal stream cues with varying fusion stages. To demonstrate it, we ran an additional experiment where we extended DeepFTSG-2 with an additional streaming cue having infrared information and named it DeepFTSG-3. Instead of two streams, DeepFTSG-3 has three streams, where the first stream input is an RGB frame (VIS), the second stream is infrared (IR) information of that frame, where we used SE-ResNet-50 backbone, and the third stream is a combination of BGS and flux for both RGB and infrared cues, making the third stream having four channels as input (1stchannel: BGS of RGB frame, 2nd-channel: flux of RGB frame, 3rd-channel: BGS of the infrared frame, 4th-channel: flux of infrared frame). In infrared, the non-visible heat radiation emitted or reflected by all objects, regardless of lighting conditions, can be imaged. Hence, infrared information provides a superior advantage in challenging conditions, such as low light, night-time, shadows, visual obstructions, degraded visual environments, and camouflaging foliage.

![](/figures/generalization.PNG)

## Comparison Results
The comparison results of proposed DeepFTSG with other methods are shown table below.

![](/figures/comparison.png)


</br>

# Video Demo of DeepFTSG-2

DeepFTSG-2 results on CDnet-2014, SBI-2015, and LASIESTA datasets, where SBI-2015 and LASIESTA are completely unseen datasets that are used for generalization purposes. The red color states missed detection (false negative), the blue color states over detection (false positive), the white color states correct foreground detection (true positive), the black color states correct background detection (true negative), the gray color states don't care regions, and dark-gray color states out of Region of Interest (ROI). 

[![Demo DeepFTSG-2](/figures/DeepFTSG-2.gif)](https://youtu.be/kdDxea5xalU)

<i>click to see the full video demo</i>

</br>

# Pre-trained weights of DeepFTSG
If you want to use pre-trained weights, put them inside **src/models/** folder.

```DeepFTSG_1.pt``` is an early fusion (appearance + BGS + Flux (motion cues)) single stream trained network model using data from CDnet-2014.

Link to download [**DeepFTSG-1 weights**](https://meru.rnet.missouri.edu/~grzc7/DeepFTSG_weights/DeepFTSG_1.pt)

```DeepFTSG_2.pt``` is an early (BGS + Flux (motion cues)) and middle (appearance + motion cues) fusion multiple stream trained network model using data from CDnet-2014.

Link to download [**DeepFTSG-2 weights**](https://meru.rnet.missouri.edu/~grzc7/DeepFTSG_weights/DeepFTSG_2.pt)

</br>

# How to use DeepFTSG

**src** folder contains all scripts used to train models, extract masks from trained models, and post-processing the output results to get labeled masks.

**weights** folder contains pre-trained weights of the DeepFTSG models, if you want to use pre-trained weights, put them inside **src/weights/** folder.

</br>

## Project Collaborators and Contact

**Author:** Gani Rahmon, and Kannappan Palaniappan

Copyright &copy; 2023-2024. Gani Rahmon and Prof. K. Palaniappan and Curators of the University of Missouri, a public corporation. All Rights Reserved.

**Created by:** Ph.D. student: Gani Rahmon  
Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Gani Rahmon**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
grzc7@mail.missouri.edu  

* **Dr. K. Palaniappan**  
205 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
palaniappank@missouri.edu

</br>

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{gani2021MUNet,
  title={Motion U-Net: Multi-cue Encoder-Decoder Network for Motion Segmentation}, 
  author={Rahmon, Gani and Bunyak, Filiz and Seetharaman, Guna and Palaniappan, Kannappan},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  pages={8125-8132},
  year={2021}
}

@article{Rahmon2023,
  title={{DeepFTSG: Multi-stream Asymmetric USE-Net Trellis Encoders with Shared Decoder Feature Fusion Architecture for Video Motion Segmentation}},
  author={Rahmon, Gani and Palaniappan, Kannappan and Toubal, Imad Eddine and Bunyak, Filiz and Rao, Raghuveer and Seetharaman, Guna},
  journal={International Journal of Computer Vision},
  year={2023},
  month={Oct},
  day={17},
  issn={1573-1405}
}
```

## ✏️ Citation

Site Change Detection (CDnet) papers if you use CDNet data in your project:

```
@inproceedings{CD2014,
    author      = "Wang, Y. and Jodoin, P. M. and Porikli, F. and Konrad, J. and Benezeth, Y. and Ishwar, P.",
    title       = "{CDnet-2014: An Expanded Change Detection Benchmark Dataset}",
    booktitle   = "IEEE Conf. CVPR Workshops",
    year        = "2014"
}
 
@inproceedings{CD2012,
    author      = "Goyette, N. and Jodoin, P. M.  and Porikli, F. and Konrad, J. and Ishwar, P.",
    title       = "{Changedetection.Net: A new change detection benchmark dataset}",
    booktitle   = "IEEE Conf. CVPR Workshops",
    year        = "2012"
} 
```