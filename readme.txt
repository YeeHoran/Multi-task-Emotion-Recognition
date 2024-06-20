Multi-task Emotion Recognition
It designs a multitask emotion recognition model that combines Valence, Arousal, Dominance (VAD) three-dimensional continuous emotion analysis and discrete emotion classification. The keypoint is that it utilizes the correlation between VAD and emotion category on a facial expression image, for multi-task joint learning and establishes constraints between emotion categories and the VAD multi-dimensional emotion space, and finally improves the recognition performance.

Code introduction: The entry is "mainpro_FER.py", and it includes data loading, training, public test, and private test. The results are output to 4 csv files and 1 txt file. The csv files are "AccProcess_classify.csv", "AccProcess_regressV.csv", "AccProcess_regressA.csv", and "AccProcess_regressD.csv". The regression loss for VAD, and the classification accuracy for Emotion category are saved in the csv files, with each row representing that epho's result. By the way, all training, public test, and private test results are saved in each row. Then, a graph can be generated using the excel tools to see the trends explicitly.

The txt file is named as "data.txt", which save the running process sequencially corresponding to each epoch. For each epoch, the content includes the results for traing, public test, and private test as well. And the final chosen model and corresponding epoch are saved as the last several lines in the file.

For the best trained models mentioned above, they are saved under "./FER2013_ResNet18RegressionTwoOutputs/" as 4 "***.t7" files, with each of them corresponding to best prediction modelsV, A, D, and Emotion category respectively. Please select the "private.t7" models. And then, these 4 models could be applied for VAD and category emotion recognition in applications.


visualize.py now can predict emotion category, VAD information together.


This version uses the complete VAD annotated FER2013 with ~2.8w train images, ~3k public test images, and ~3k private test images. It has removed the images which are not face looking.
The dataset is in data.rar, which is required to be depressed before running the mainFER.py.


Updated on June 20th.
