# ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls
the code for paper: ResNet-based Bio-acoustics Presence Detection Technology of Hainan Gibbon Calls

This repository only contains the code and the best pre-training model in this paper. If you want to download the corresponding data set, please go to the following link  
https://doi.org/10.5281/zenodo.3991714

Or you can get the complete code and all the data used in this paper from this link  
link：https://pan.baidu.com/s/10zfY0s2wuFRDbPiomDKIbA   
password：szku 

## Getting start:
step1: download the dataset from: https://doi.org/10.5281/zenodo.3991714.  
step2: move the corresponding data into the "Data" folder.  
step3: run the script Extract_Feature.py in "Extract_Feature" to extract the mel spectrogram from traning raw data.  
step4: run the script trainging to train your own BPDnet model.          
step5: run the script evaluation.py and Every_Result.py in "evaluation" folder to test you model, you well get classification evaluation results of each test file. 

if you just want to use our pre-training model to get a detect of Hainan gibbon, please follow this:  
step1: fill you own 8 hours wav files in 'Data/Raw_Data/Test'.  
step2: run the script evaluation.py in "evaluation" folder.    
you will get the excel file in "BPDnet_Result", that can tell you which ten seconds are there have Hainan gibbon call.   
 
## BPDnet structure
![image](https://github.com/ruwenda/ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls/blob/master/pict/network1.png)
![image](https://github.com/ruwenda/ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls/blob/master/pict/network2.png)

## The performance of the BPDnet
![image](https://github.com/ruwenda/ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls/blob/master/pict/res1.png)

The baseline paper is: "Automated detection of Hainan gibbon calls for passive acoustic monitoring" (Remote Sensing in Ecology and Conservation).

## The performance of the augmentation and post-processing
![image](https://github.com/ruwenda/ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls/blob/master/pict/res2.png)
![image](https://github.com/ruwenda/ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls/blob/master/pict/res3.png)
![image](https://github.com/ruwenda/ResNet-based-Bio-acoustics-Presence-Detection-Technology-of-Hainan-Gibbon-Calls/blob/master/pict/res4.png)
