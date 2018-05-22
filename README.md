# Audio Chime
Paper Implementation for:

A convolutional neural network approach for acoustic scene classification [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7966035)]

We have worked on the **[CHIME](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging)** Audio Dataset for Audio tagging. We train on `48KHz` and test on `16KHz` Audio.

## About the Dataset
The annotations are based on a set of 7 label classes. For each chunk, multi-label annotations were first obtained for each of 3 annotators. There are 1946 such 'strong agreement' chunks is the development dataset, and 816 such 'strong agreement' chunks in the evaluation dataset.

## Cloning the repo
Go ahead and clone this repository using
```
$ git clone https://github.com/DeepLearn-lab/audio_CHIME.git
``` 

## Quick Run
If you are looking for a quick running version go inside `single_file` folder and run
```
$ python mainfile.py
```

## Detailed Task
The process involves three steps:
1. Feature Extraction
2. Training on Development Dataset
3. Testing on Evaluation Dataset

### Feature Extraction

We are going to extract **mel frequencies** on raw audio waveforms. Go ahead and uncomment  
```feature_extraction``` function which would extract these features and save it in the `.f` pickle.

### Training

We train our model on these extracted featuers. We use a convolution neural network for training and testing purpose. Alteration in model can be done in `model.py` file.
All `hyper-parameters` can be set in `util.py`. Once you have made all the required changes or want to run on the pre-set ones, run 
```
$ python mainfile.py 
```

This will run the model which we test and use `EER` for rating our model.
