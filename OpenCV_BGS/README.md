# Running OpenCV Background Subtraction (BGS)

**To get BGS results for use in DeepFTSG-1 and DeepFTSG-2**

1. Change the input/output paths and image file format in ```config.txt``` file accordingly. 
```
# Config file to run OpenCV Background Subtraction

##### IO Parameters #####
# Input sequence path (give full path)
input_dir = /mnt/d/GitHub/DeepFTSG-IJCV/src/datasets/test/CD2014/DATASET/baseline/highway/input 

# Input image file format (e.g., jpg, png)
image_ext = jpg

# Ouput path (give full path)
output_dir =  /mnt/d/GitHub/DeepFTSG-IJCV/src/datasets/test/CD2014/BGS/baseline/highway/ 
``` 

2. Create a ```build``` folder:  
```
mkdir build
```

3. Enter the ```build``` folder:
```
cd build
```

4. Run ```cmake```:
```
cmake ..
```

5. Run ```make```:
```
make
```

6. Go to ```bin/linux``` folder:
```
cd ../bin/linux
```

7. Run ```BGSubOpenCV```:
```
./BGSubOpenCV
```

8. The output of BGS will be saved in the provided output path.

## ✏️ Citation

Site OpenCV background subraction papers if you use OpenCV_BGS code in your project:

```
@article{Zivkovic,
    title       = "{Efficient adaptive density estimation per image pixel for the task of background subtraction}",
    journal     = "Pattern Recognition Letters",
    volume      = "27",
    number      = "7",
    pages       = "773 - 780",
    year        = "2006",
    author      = "Zivkovic, Z. and van der Heijden, F."
}

@inproceedings{Zivkovic2,
  author        = "Zivkovic, Z.",
  booktitle     = "Int. Conf. Pattern Recognition", 
  title         = "{Improved adaptive Gaussian mixture model for background subtraction}", 
  year          = "2004",
  volume        = "2",
  pages         = "28-31"
}
```