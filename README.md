Music Genre Classification
==============================

[![pypi](https://img.shields.io/pypi/v/cookiecutter.svg)](https://pypi.python.org/pypi/cookiecutter)
[![python](https://img.shields.io/pypi/pyversions/cookiecutter.svg)](https://pypi.python.org/pypi/cookiecutter)


Music Genre Classification using Million Song Dataset


## Problem Statement

Many Music Information Retrival (MIR) research have been made on the subject of music genre classification. Each study with a different approach as to which acoustic features and which algorithms to base the classification upon and which tracks to use for the evaluation of the results. Because of this, the outcomes differ substanttially.

A study on genre classification using Million Song Dataset (MSD) has been implemented here. Music genre classification is always challenging as it is sometimes impossible for a human to find the genre of a music track. The purpose of this project is to classifiy music genre using music features derived from an acoustic analysis.

We know that the low-level feature MFCC, or the corresponding timbre feature, will be a good basis for the classification since it is a feature that mostly used. But adding high level features like tempo, loudness, key and pitch in combination may improve the results because those features vary among different genres.

In machine learning perspective prediction is a critical task in imbalanced data multi-class classification. 

## Dataset

The dataset we have chosen in this study is the Million Song Dataset (MSD) which has been downloaded from thee link http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_genre_dataset.zip


This dataset is a subset of the MSD which contains only 59,600 songs with the following information:

- Song ID, title, artist name

- Genre - 10 categories: classic pop and rock, folk, dance and electronica, jazz and blues, soul and reggae, punk, metal, classical, pop, hip-hop

- Loudness (numerical, from -40 to 0): In musical terminology, loudness is the ”quality of a sound that is the pri- mary psychological correlate of physical strength”. The loudness feature is an overall estimation of the track’s loudness in decibel (dB).

- Tempo (numerical, from 0 to 255): Tempo is the speed or pace of a given track measured in beats per minute (BPM). As the tempo varies during the track, the tempo feature is an overall estimation of the track’s tempo.

- Time signature (categorical, from 0 to 7)

- Key (categorical, from 0 to 11): The term key can be used in many different ways. In this case the meaning of the term key is the tonic triad, the final point of rest of a track

- Mode (binary, 0 or 1)

- Duration (numerical)

- Average and variance of timbre vectors (numerical) – 12 variables for each (24 in total)

## Data Visualization

Let us explore the dataset to investigate if there could be any evidence of relationship between `genre` and other features. First we can find the distribution of all 10 genre classes and it is seen that the distribution is pretty imbalanced (Figure below). 
![alt text](https://github.com/raktim314/music_genre_classification/blob/master/class_counts.png)
| Class  |Count   |Percentage  |
|---|---|---|
|classic pop and rock |23895   | 40.092  |
|punk |3200   |5.369   |
|folk |13192 |22.134
|pop  |1617   |2.713   |
|dance and electronica   |2103   |3.529   |
|metal   |2103   |3.529   |
|jazz and blues   |4334   |7.272   |
|classical   |1874   |3.144   |
|hip-hop   | 434  |0.728   |
|soul and reggae   |4016   |6.738   |

As the class is imbalanced, it may be a challenge to fit ML model.

In the boxplot below we can see relationship between different genres and their loudness and tempo.
![alt text](https://github.com/raktim314/music_genre_classification/blob/master/loudness.png)

![alt text](https://github.com/raktim314/music_genre_classification/blob/master/tempo.png)

Though there are some outliers in `loudness` and `tempo` of each genres, we are not going to remove them because in practical world these can go beyond the average limits.

## Modelling

I choose 20% of the total dataset as test set for the models.

The results of the baseline classification models are shown as below:
|  Model | Accuracy using Cross Validation  | Accuracy on test set
|---|---|---|
| Logistic Regression  |54.9%   | 55.0%  |
|KNN   |43.6%   |43.3%   |
|Random Forest Classifier   |55.8%   |55.4%   |
|Gradient Boosting Classifier   |58.6%   |58.4%   |

![alt text](https://github.com/raktim314/music_genre_classification/blob/master/confusion-matrix.png)


After applying grid search for hyperparameter optimization on Gradiant Boosting Classifiers, the classification only achieved 62% accuracy and F1-Score of 60% at best, which we believe is a rather poor, but a reliable result.

Also, from the confusion matrix above, it is seen that **hip-hop** and **pop** music are badly misclassified though other genres are well classified.


 Using this classifier in practice in an application would lead to a lot of misclassified tracks. The main reasons for the not so successful results are:

- The small dataset
- The genres chosen - some of them sound similar
- The selection and usage of the features

The choice of dataset, genres, features, software and algorithms is, however, well motivated which contributes to the reliability of the results. In particular, the choice of genres and the large dataset reflect realistic conditions for classification of music tracks.

It is an essential problem of choosing the dataset to obtain a reliable classification. The dataset in this study was chosen because of the connection to a real well-known music situation. The purpose was not to evaluate how good an classifier could be when running on a certain dataset. The purpose was to be able to classify any kind of dataset which will lead us for further research. Implementing deep learning algorithms on dataset like **GTZAN** often gives more accuracy in this context.