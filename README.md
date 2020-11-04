# Generate Music based on Emotion

~~TODO: get TOC from https://ecotrust-canada.github.io/markdown-toc/~~ 
## Overview

## Requirement

To run Step 3-1 and 3-2, MusicVAE is required. [Here](https://github.com/magenta/magenta/blob/master/README.md#installation) is the description of how to install it.

It also need python library 

## Data
Data and external lib can be download from [oneDrive].
#### Dataset
##### 0.DEAM

The [DEAM dataset](http://cvml.unige.ch/databases/DEAM/). Only including the MP3 audio files and the dynamic per-half-second annotations.

It also have 2 folders, one called 'MIDI' is storing the MIDI format of MP3 files, which convert by the online converter [bearaudiotool](https://www.bearaudiotool.com/MP3-to-MIDI). one called 'WAV'is stroing the 'WAV' format of MP3 files which convert by FFmpeg.

##### 1. 400-100 dataset

The output by Step 1.

##### 2. Best model

The best model we found by grid search in Step 2.

##### 3-1 Music VAE checkpoint

The checkpoint (training model) of [MusicVAE](https://github.com/magenta/magenta/tree/master/magenta/models/music_vae).

##### 3-2 Output from MusicVAE

The interpolation output by MusicVAE.

##### 3-3 Emotion Changing result

The re-evaluation result of the output in step 3-2 by the best model we found in Step 2.

#### External lib

##### FFmpeg

This is for converting MP3 file to WAV file.

##### TiMIDIty

This is for converting MIDI file to WAV file.

## Details of each step

### Step 1 Extra Feature

Firstly, we should explain the words we will used in the next:
1. **'emotion label'**: a $1\times 2$ vector, include 2 real numbers. One for the value of Arousal and another for the value of Valence in the V-A emotion model.
2. **'data'**: one music song in the dataset, should be more than 45 seconds long, with 60 emotion labels.
3. **'fragment'**: one data have 60 half-second fragment. Each fragment corresponds to one emotion label.
4. **'fragment size'**: it depends on the sampling frequency and the number of features exacted in one sampling.

Three things we do in this step:

Firstly, we check and find all data which is available on both format and randomly sample 400 data for training and 100 data for testing. We name them as "400-100 dataset". "Available" means the data have a proper length in piano roll for MIDI (piano roll is around 4500 long) and more than 45 seconds for WAV.

Secondly, we exact features from the data. We create 4 datasets in total, size of **each data** are showed below:

|         | wav                          | midi                      |
| ------- | ---------------------------- | ------------------------- |
| CNN     | 60fragment\*50sf\*128feature | 60frame\*50sf\*128feature |
| CNN+RNN | 60fragment\*1sf\*128feature  | 60frame\*1sf\*128feature  |

The 'sf' means 'sampling frequency'. For CNN, it is 50 and for CNN+RNN it is 1.

In CNN model, we assume fragments in one data are **not related to each other**, so we have $500\times 60=3000$ fragments in total, and now 'fragment' equals 'data' in the usual sense. In CNN+RNN model, we ignore the dimension 'sf' because it equals 1.

Thus we get 4 datasets in the end, 2 of the dataset shape is **3000fragment\*50\*128**, and 2 of the dataset shape is **500data\*60\*128**.

Finally, we save all the datasets.

### Step 2 Find the best CNN/CNN+RNN model

#### 2.1 CNN

We just set a CNN network and train it on the data we get from step1.

The aim is to know which music format is better - WAV or converted MIDI.

Input shape is 50\*128.

The architect is:
Conv2D(64,3,relu) ->Conv2D(64,3,relu) ->maxpooling(2);

then it split into two branch for valence and arousal. For each branch:

Conv2D(64,3,relu) ->Conv2D(64,3,relu) ->maxpooling(2)->Conv2D(128,3,relu)->maxpooling(2)  ->dropout(0.25  dropped)  ->Conv2D(256,3,relu)  ->max-pooling(2) ->dropout(0.25) ->Conv2D(256,3,relu) ->maxpooling(2) ->dropout(0.25)->Conv2D(256,3,relu) ->maxpooling(1,3) ->dropout(0.25) ->flatten ->dense(256)->dropout(0.5) ->dense(256) ->dropout(0.5) ->dense(1)

Output shape is 1\*2 for valence and arousal.

#### 2.2 CNN+RNN

We do a simple CNN first and then run a RNN(bi-gru). We using the grid search to find the best model. And will do on both WAV and MIDI files.

Input shape is 60\*128.

The architect is:
Conv2D(**cf**,3,relu) ->batchNormalization ->reshape; then it split into two branch for valence and arousal.dense(**vaDense**) ->Bi-GRU(**Gru**) ->Dense(1) ->flatten

Output shape is 60*2.

The cf, vaDense and Gru are the hyper-parameter we will decide by grid search.

The range for CNN+RNN grid search is:
cf: 8,16
vaDense: 8, 16
Gru: 8,16,32
batch_size: 5,10,15.

#### Compare model

Every models have 2 datasets in different music format, thus we have 4 trainings at all.

The loss is the MSE for CNN+RNN. And we find the best model in CNN+RNN is when format is WAV, cf=16, vaDense=8, Gru = 32, batch_size = 15.

### Step 3
#### 3.1 Train MusicVAE

Firstly we build the dataset for MusicVAE, and then train it.

Refereance: [MusicVAE-training-your-own-musicvae](https://github.com/magenta/magenta/tree/master/magenta/models/music_vae/#training-your-own-musicvae).

#### 3.2 Interpolate and Evaluate the result

In this part, we randomly sample some fragments of same data.

Because MusicVAE requires the format of MIDI, so we sample MIDI data. We also write a function on split MIDI by time.

Then we put 2 sampled data into MusicVAE and interpolate to create 2 outputs.

And we convert the outputs into WAV format because our best emotion model is CNN+RNN on WAV format.

Finally we record the emotion labels of our result and the original emotion labels of 2 sampled data.

