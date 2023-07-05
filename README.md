# Speech Emotion Analyzer

The idea behind creating this project was to build a machine learning model that could detect emotions from the speech we have with each other all the time. Nowadays personalization is something that is needed in all the things we experience everyday.

So why not have a emotion detector that will guage your emotions and in the future recommend you different things based on your mood. This can be used by multiple industries to offer different services like marketing company suggesting you to buy products based on your emotions, automotive industry can detect the persons emotions and adjust the speed of autonomous cars as required to avoid any collisions etc.

## Datasets:
Made use of the ravdess dataset:

RAVDESS. This dataset includes around 1500 audio file input from 24 different actors. 12 male and 12 female where these actors record short audios in 8 different emotions i.e 1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised.
Each audio file is named in such a way that the 7th character is consistent with the different emotions that they represent.

## Feature Extraction
The next step involves extracting the features from the audio files which will help our model learn between these audio files. For feature extraction we make use of the LibROSA library in python which is one of the libraries used for audio analysis.

![image](https://github.com/chandra237/Deployment/assets/125145475/63bd10e5-ff8f-4c80-add0-fffc5b3a2efd)

Here there are some things to note. While extracting the features, all the audio files have been timed for 3 seconds to get equal number of features.

The sampling rate of each file is doubled keeping sampling frequency constant to get more features which will help classify the audio file when the size of dataset is small.

The extracted features are array of values with lables appended to them.

## Building Models
Since the project is a classification problem, we go with Supervised Machine learning(SVM). We also built Multilayer perceptrons and Long Short Term Memory models but they under-performed with very low accuracies which couldn't pass the test while predicting the right emotions.

Building and tuning a model is a very time consuming process. The idea is to always start small without adding too many layers just for the sake of making it complex. After testing out with layers, the model which gave the max validation accuracy against test data was little more than 70%

## Testing out with live voices.
In order to test out our model on voices that were completely different than what we have in our training and test data, we recorded our own voices with dfferent emotions and predicted the outcomes. You can see the results below: The audio contained a male voice which said "This coffee sucks" in a angry tone.


NOTE: If you are using the model directly and want to decode the output ranging from 0 to 9 then the following list will help you.

0 - Happy <br>
1 - Neutral <br>
2 - Sad <br>
3 - Angry <br>

## Conclusion
Building the model was a challenging task as it involved lot of trail and error methods, tuning etc. The model is very well trained to distinguish between voices and it distinguishes with 100% accuracy. The model was tuned to detect emotions with more than 70% accuracy. Accuracy can be increased by including more audio files for training.
