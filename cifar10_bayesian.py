import pickle
import numpy as np
import time as time
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

testdict = unpickle('./cifar-10-batches-py/test_batch')
datadict1 = unpickle('./cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('./cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('./cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('./cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('./cifar-10-batches-py/data_batch_5')
labeldict = unpickle('./cifar-10-batches-py/batches.meta')

X1 = datadict1['data']
X2 = datadict2['data']
X3 = datadict3['data']
X4 = datadict4['data']
X5 = datadict5['data']
Y1 = datadict1['labels']
Y2 = datadict2['labels']
Y3 = datadict3['labels']
Y4 = datadict4['labels']
Y5 = datadict5['labels']

X1 = X1.reshape(10000, 3, 32, 32).astype("int")
X2 = X2.reshape(10000, 3, 32, 32).astype("int")
X3 = X3.reshape(10000, 3, 32, 32).astype("int")
X4 = X4.reshape(10000, 3, 32, 32).astype("int")
X5 = X5.reshape(10000, 3, 32, 32).astype("int")

testDataArray = testdict["data"]
testLabelArray = testdict["labels"]

testDataArray = testDataArray.reshape(10000, 3, 32, 32).astype("int")

dataArray = np.concatenate([X1, X2])
dataArray = np.concatenate([dataArray, X3])
dataArray = np.concatenate([dataArray, X4])
dataArray = np.concatenate([dataArray, X5])

printDataArray = dataArray.transpose(0, 2, 3, 1)

labelArray = np.concatenate([Y1, Y2])
labelArray = np.concatenate([labelArray, Y3])
labelArray = np.concatenate([labelArray, Y4])
labelArray = np.concatenate([labelArray, Y5])

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
labelNamesArray = labeldict["label_names"]

testDataArray = np.array(testDataArray)
testLabelArray = np.array(testLabelArray)
dataArray = np.array(dataArray)
labelArray = np.array(labelArray)

print(dataArray.shape)

def cifar10_color(dataArray):
    dataArray_mean = np.zeros([50000, 3])
    for pic in range(0, dataArray.shape[0]):
        for colorLayer in range(0, dataArray[pic].shape[0]):
            dataArray_mean[pic][colorLayer] = np.mean(dataArray[pic][colorLayer])
    return dataArray_mean

def cifar10_naivebayes_learn(dataArray_mean, labelArray):
    parametersArray = np.zeros([10, 7])
    for i in range(0, 10):
        tempArray = []
        for j in range(0, dataArray_mean.shape[0]):
            if (i == labelArray[j]):
                tempArray.append(dataArray_mean[j])
        tempArray = np.array(tempArray)
        tempArray = np.transpose(tempArray, (1, 0))
        parametersArray[i][0] = np.mean(tempArray[0])
        parametersArray[i][1] = np.mean(tempArray[1])
        parametersArray[i][2] = np.mean(tempArray[2])
        parametersArray[i][3] = np.std(tempArray[0])
        parametersArray[i][4] = np.std(tempArray[1])
        parametersArray[i][5] = np.std(tempArray[2])
        parametersArray[i][6] = 0.1
    return parametersArray

def cifar10_classifier_naivebayes(picture, parametersArray):
    pictureColors = np.zeros(3)
    for i in range(0, 3):
        pictureColors[i] = np.mean(picture[i])
    label = 10
    labelscore = 0
    for i in range(0, parametersArray.shape[0]):
        score1 = np.divide(norm.pdf(np.divide(np.subtract(pictureColors[0], parametersArray[i][0]), parametersArray[i][3])), parametersArray[i][3])
        score2 = np.divide(norm.pdf(np.divide(np.subtract(pictureColors[1], parametersArray[i][1]), parametersArray[i][4])), parametersArray[i][4])
        score3 = np.divide(norm.pdf(np.divide(np.subtract(pictureColors[2], parametersArray[i][2]), parametersArray[i][5])), parametersArray[i][5])
        score = score1 * score2 * score3 * parametersArray[i][6]
        if(score > labelscore):
            labelscore = score
            label = i
    return label

def cifar10_bayes_learn(dataArray_mean, labelArray):
    parametersArray2 = np.zeros([10, 4])
    covMatrices = []
    for i in range(0, 10):
        tempArray = []
        for j in range(0, dataArray_mean.shape[0]):
            if (i == labelArray[j]):
                tempArray.append(dataArray_mean[j])
        tempArray = np.array(tempArray)
        tempArray = np.transpose(tempArray, (1, 0))
        parametersArray2[i][0] = np.mean(tempArray[0])
        parametersArray2[i][1] = np.mean(tempArray[1])
        parametersArray2[i][2] = np.mean(tempArray[2])
        covMatrices.append(np.cov([tempArray[0], tempArray[1], tempArray[2]]))
        parametersArray2[i][3] = 0.1
    return parametersArray2, covMatrices

def cifar10_classifier_bayes(picture, parametersArray2, covMatrices):
    pictureColors = np.zeros(3)
    for i in range(0, 3):
        pictureColors[i] = np.mean(picture[i])
    label = 10
    labelscore = 0
 
    for i in range(0, parametersArray2.shape[0]):
        score = multivariate_normal.pdf(pictureColors, mean=[parametersArray2[i][0], parametersArray2[i][1], parametersArray2[i][2]], cov=covMatrices[i])
        score = score * parametersArray2[i][3]
        if(score > labelscore):
            labelscore = score
            label = i
    return label

dataArray_mean = cifar10_color(dataArray)
parametersArray = cifar10_naivebayes_learn(dataArray_mean, labelArray)
parametersArray2, covMatrices = cifar10_bayes_learn(dataArray_mean, labelArray)
naiveCorrectPredictions = 0
betterCorrectPredictions = 0
totalPredictions = 0

for i in range(0, testDataArray.shape[0]):
    plt.figure(1)
    plt.clf()
    totalPredictions += 1
    start = time.time()
    naivePrediction = cifar10_classifier_naivebayes(testDataArray[i], parametersArray)
    betterPrediction = cifar10_classifier_bayes(testDataArray[i], parametersArray2, covMatrices)
    stop = time.time()
 
    if(naivePrediction == testLabelArray[i]):
        naiveCorrectPredictions += 1
    print(str(naiveCorrectPredictions / totalPredictions * 100) + ' %')
    
    if(betterPrediction == testLabelArray[i]):
        betterCorrectPredictions += 1
    print(str(betterCorrectPredictions / totalPredictions * 100) + ' %')
    '''
    accRating = naiveCorrectPredictions / totalPredictions * 100
    accRating2 = betterCorrectPredictions / totalPredictions * 100
    plt.text(37, 5,'Naive\nClassifier\nCurrent\nAccuracy:\n ' + str(round(accRating, 3)) + '%' , ha="center", va="center")
    plt.text(37, 20,'Better\nClassifier\nCurrent\nAccuracy:\n ' + str(round(accRating2, 3)) + '%' , ha="center", va="center")
    plt.text(5, -3,'naive bayes prediction: ' + labelNamesArray[naivePrediction], ha="center", va="center")
    plt.text(28, -3,'better bayes prediction: ' + labelNamesArray[betterPrediction], ha="center", va="center")
    plt.text(2, 34,'images done: ' + str(i+1), ha="center", va="center")
    plt.title(f"Image {i} label={labelNamesArray[testLabelArray[i]]} (num {labelArray[i]})")
    plt.text(20, 34,'computing time: ' + str(round(stop - start, 3)) + 's', ha="center", va="center")
    plt.imshow(printDataArray[i])
    plt.pause(0.1)
    '''
print('total accuracy naive: ')
print(str(naiveCorrectPredictions / totalPredictions * 100) + ' %')
print('total accuracy better: ')
print(str(betterCorrectPredictions / totalPredictions * 100) + ' %')
