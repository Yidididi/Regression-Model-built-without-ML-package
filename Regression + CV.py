# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:44:12 2019

@author: Yidi Kang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ReadFile(x):
    return pd.read_csv(x,sep=",")

train_100_10 = ReadFile("train-100-10.csv")
test_100_10 = ReadFile("test-100-10.csv")
train_100_100 = ReadFile("train-100-100.csv")
test_100_100 = ReadFile("test-100-100.csv")
train_1000_100 = ReadFile("train-1000-100.csv")
test_1000_100 = ReadFile("test-1000-100.csv")

train_50 = open("train-50(1000)-100.csv","w")
train_100 = open("train-100(1000)-100.csv","w")
train_150 = open("train-150(1000)-100.csv","w")

df_50 = train_1000_100[0:50]
df_50.to_csv("train-50(1000)-100.csv",index=True,sep=",")
df_100 = train_1000_100[0:100]
df_100.to_csv("train-100(1000)-100.csv",index=True,sep=",")
df_150 = train_1000_100[0:150]
df_150.to_csv("train-150(1000)-100.csv",index=True,sep=",")


# Regularization
def Calculatex(data):
    x = [[1]*(data.columns.size) for i in range(len(data))]
    i = 0
    for i in range(len(data)):
        for j in range(1,(data.columns.size)):
            x[i][j]=data.iloc[i][j-1]
    x = np.asmatrix(x)
    return x
def Calculatey(data):
    y = np.asmatrix([[0] for i in range(len(data))])
    data = data.reset_index()
    for i in range(len(data)):
        y[i] = data["y"][i]
    return y

def Calculatew(l,x,y,data):
    Ix = np.identity( np.shape(x)[1] )  # Create identity matrix
    w = (np.linalg.inv((x.T * x) + (l*Ix))) * (x.T *y)
    return w

def CalculateTrainMSE(lamlist,data):
    MSE = [0 for i in lamlist]
    w = [0 for i in lamlist]
    xtrain = Calculatex(data)
    ytrain = Calculatey(data)
    for a in lamlist:
        l = a
        w[a] = Calculatew(l,xtrain,ytrain,data)
        MSE[a] = np.sum(np.square(xtrain*w[a]-ytrain))/np.shape(xtrain)[0]
    return (MSE,w,xtrain,ytrain)

def CalculateTestMSE(lamlist,testdata,wtrain):
    MSE = [0 for a in lamlist]
    x = Calculatex(testdata)
    y = Calculatey(testdata)
    for a in lamlist:
        w = wtrain[a]
        MSE[a] = np.sum(np.square(x*w-y))/np.shape(x)[0]
    return MSE

train_100_10Final = CalculateTrainMSE(range(150),train_100_10)
train_100_10MSE = train_100_10Final[0]
train_100_10w = train_100_10Final[1]
test_100_10MSE = CalculateTestMSE(range(150),test_100_10,train_100_10w)

train_100_100Final = CalculateTrainMSE(range(150),train_100_100)
train_100_100MSE = train_100_100Final[0]
train_100_100w = train_100_100Final[1]
test_100_100MSE = CalculateTestMSE(range(150),test_100_100,train_100_100w)

train_1000_100Final = CalculateTrainMSE(range(150),train_1000_100)
train_1000_100MSE = train_1000_100Final[0]
train_1000_100w = train_1000_100Final[1]
test_1000_100MSE = CalculateTestMSE(range(150),test_1000_100,train_1000_100w)

train_50_1000_100Final = CalculateTrainMSE(range(150),df_50)
train_50_1000_100MSE = train_50_1000_100Final[0]
train_50_1000_100w = train_50_1000_100Final[1]
test_50_1000_100MSE = CalculateTestMSE(range(150),test_1000_100,train_50_1000_100w)

train_100_1000_100Final = CalculateTrainMSE(range(150),df_100)
train_100_1000_100MSE = train_100_1000_100Final[0]
train_100_1000_100w = train_100_1000_100Final[1]
test_100_1000_100MSE = CalculateTestMSE(range(150),test_1000_100,train_100_1000_100w)

train_150_1000_100Final = CalculateTrainMSE(range(150),df_150)
train_150_1000_100MSE = train_150_1000_100Final[0]
train_150_1000_100w = train_150_1000_100Final[1]
test_150_1000_100MSE = CalculateTestMSE(range(150),test_1000_100,train_150_1000_100w)

# 6 plot of MSE for 6 datasets
def plotMSE(title,lamlist,trainMSE,testMSE):
    plt.plot(lamlist,trainMSE,color="blue")
    plt.plot(lamlist,testMSE,color="red")
    plt.title(title)
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    

p100_10 = plotMSE("MSE Plot for 100_10 dataset",range(150),train_100_10MSE,test_100_10MSE)
p100_100 = plotMSE("MSE Plot for 100_100 dataset",range(150),train_100_100MSE,test_100_100MSE)
p1000_100 = plotMSE("MSE Plot for 1000_100 dataset",range(150),train_1000_100MSE,test_1000_100MSE)

# find lambda which give the least test set MSE
dataset = [test_100_10MSE,test_100_100MSE,test_1000_100MSE,test_50_1000_100MSE,test_100_1000_100MSE,test_150_1000_100MSE]
title = ["100_10","100_100","1000_100","50_1000_100","100_1000_100","150_1000_100"]
for i in dataset:
    print("For Dataset",title[dataset.index(i)],", lambda=",i.index(np.min(i)),"gives the least MSE value",round(np.min(i),2))

# plot lambda~(1,150) for 3 datasets
p100_100_2 = plotMSE("149 MSE Plot for 100_100 dataset",range(1,150),train_100_100MSE[1:150],test_100_100MSE[1:150])
p_50_1000_100_2 = plotMSE("149 MSE Plot for 50_1000_10 dataset",range(1,150),train_50_1000_100MSE[1:150],test_50_1000_100MSE[1:150])
p_100_1000_100_2 = plotMSE("149 MSE Plot for 100_1000_100 dataset",range(1,150),train_100_1000_100MSE[1:150],test_100_1000_100MSE[1:150])

# 
# When lambda = 0, the model is overfitting. At this time, there are too many pramaters, 
# the models focus on fitting training data but not linear regression, 
# so when applying the model to test data,it will cause abnormally high variance.

# Cross Validation:
k = 10
a = 0
l = 1
lamlist = range(10)
k = 3
def CV(lamlist,k,data):
    MSE = [0 for a in lamlist]
    for a in lamlist:
        l = lamlist[a]
        tempMSE = 0
        for i in range(k):
            CVtest = data.loc[(len(data)/k)*i:(len(data)/k)*(i+1)-1]
            CVtrain1 = data.loc[0:(len(data)/k)*i-1]
            CVtrain2 = data.loc[(len(data)/k)*(i+1):len(data)]
            CVtrain = pd.concat([CVtrain1,CVtrain2],axis= 0) # create CV File
            # calculate
            xtrain = Calculatex(CVtrain)
            ytrain = Calculatey(CVtrain)
            wtrain = Calculatew(l,xtrain,ytrain,CVtrain)
            x = Calculatex(CVtest)
            y = Calculatey(CVtest)
            tempMSE = tempMSE + np.sum(np.square(x*wtrain-y))/np.shape(x)[0]
        aveMSE = tempMSE/k
        MSE[a] = aveMSE
    return MSE
CV_train_100_10 = CV(range(50),10,train_100_10)
CV_train_100_100 = CV(range(50),10,train_100_100)
CV_train_1000_100 = CV(range(50),10,train_1000_100)
CV_train_50_1000_100 = CV(range(50),10,df_50)
CV_train_100_1000_100 = CV(range(50),10,df_100)
CV_train_150_1000_100 = CV(range(50),10,df_150)

#
dataset = [CV_train_100_10,CV_train_100_100,CV_train_1000_100,CV_train_50_1000_100,CV_train_100_1000_100,CV_train_150_1000_100]
title = ["100_10","100_100","1000_100","50_1000_100","100_1000_100","150_1000_100"]
for i in dataset:
    print("According to CV: For Dataset",title[dataset.index(i)],", lambda=",i.index(np.min(i)),"gives the least MSE value",round(np.min(i),2))

# 
# The lambda and MSE from CV and Q2(a) varied, but not by much. MSEs are kind of close

# drawback of CV:
# The computation cost is vary high, may not possible in applcation.

# factors affacting CV:
# 1.k. How many folders we decided to split data will affact CV performance.
# 2.How we split the data may affect performance. (if data are extreme in one set)

# Learning Curve

testdata = test_1000_100
nlist = range(40,1001,20)
x = Calculatex(testdata)
y = Calculatey(testdata)

def CalculateLCMSE(data,x,y,nlist,l):
    MSE = [0 for i in range(len(nlist))]
    for i in range(len(nlist)):
        n = nlist[i] # number of subsets
        tempMSE = [0 for a in range(10)]
        for a in range(10): # find subset for 10 times
            seedlist = np.random.randint(0,len(data),n)
            df = data.loc[np.array(seedlist).tolist()]
            xtrain = Calculatex(df)
            ytrain = Calculatey(df)
            wtrain = Calculatew(l,xtrain,ytrain,df)
            tempMSE[a] = np.sum(np.square(x*wtrain-y))/np.shape(x)[0]
        MSE[i] = np.sum(tempMSE)/len(tempMSE)
    return MSE

MSE_l_1 = CalculateLCMSE(train_1000_100,x,y,nlist,1)
MSE_l_25 = CalculateLCMSE(train_1000_100,x,y,nlist,25)
MSE_l_150 = CalculateLCMSE(train_1000_100,x,y,nlist,25)

# plot learning curve
def plotLC(title,nlist,MSE):
    plt.plot(nlist,MSE,color="red")
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("sample size")
pLC_1 = plotLC("Learning curve when lambda=1",nlist,MSE_l_1)
pLC_25 = plotLC("Learning curve when lambda=25",nlist,MSE_l_25)
pLC_150 = plotLC("Learning curve when lambda=150",nlist,MSE_l_150)


train_50.close()
train_100.close()
train_150.close()










