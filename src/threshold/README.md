# Resize and Threshold output masks of DeepFTSG-1 or DeepFTSG-2

## Part 3 : Threshold

**To get binary masks ofDeepFTSG-1 or DeepFTSG-2**

1. Change ```networkName``` with the network name you want to apply threshold in ```threshold.m```.

2. Change ```orgImgFolder``` and ```maskFolder``` paths accordingly in ```threshold.m```. The example is given for CDnet dataset.

3. Change input image names and extension accordingly in ```threshold.m```

4. Change the folder path of video sequences and maximum number of frames in that sequence accordingly in ```runThreshold.m```. The example is given for CDnet dataset.

5. Run ```runThreshold.m```

This script will resize and threshold extracted masks to generate binary masks and save the binary masks inside ```output_th``` folder.  