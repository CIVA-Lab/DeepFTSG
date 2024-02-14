# Train DeepFTSG models and Extract Masks for trained / pre-trained model of DeepFTSG

## Part 1 : Train Models

**To train DeepFTSG-1 or DeepFTSG-2**

1. Put your data used to train the network in a folder called ```dataset/train/``` folder. Baseline/Highway is given from CDnet-2014 dataset as example. Please cite CDnet papers if you use Baseline/Highway in your work.

2. Background Subtraction (BGS) and flux masks are optained using traditional methods. Baseline/Highway is given from CDnet-2014 dataset as example. Please cite CDnet papers if you use Baseline/Highway in your work.

3. Change input and label paths and extensions accordingly in ```Train_DeepFTSG_1.py``` or ```Train_DeepFTSG_2.py```

4. Run ```Train_DeepFTSG_1.py``` or ```Train_DeepFTSG_2.py```

## Part 2 : Extract Masks

**To extract masks of DeepFTSG-1 or DeepFTSG-2**

1. To extract masks using trained / pre-trained models of DeepFTSG create a new folder with dataset name inside ```dataset/test/``` folder and and put your data inside created folder. Baseline/Highway is given from CDnet-2014 dataset as example. Please cite CDnet papers if you use Baseline/Highway in your work.

2. Change dataset paths and extensions accordingly in ```Infer_DeepFTSG_1.py``` or ```Infer_DeepFTSG_2.py```

3. Change video sequence paths accordingly in ```CD2014.txt``` which is located inside ```files``` folder . Some examples of video sequence taken from CDNet 2014 are given inside  ```CD2014.txt```

4. Run ```Infer_DeepFTSG_1.py``` or ```Infer_DeepFTSG_2.py```

This script will extract masks using trained / pre-trained models of DeepFTSG for the given dataset and save the result of output masks inside ```output``` folder.