import os
import pickle
import numpy

data_path="EEG_dataset/s01.dat"

with open(data_path,"rb") as f:
    subject = pickle.load(f,encoding = "latin1")
data=subject["data"]
labels=subject["labels"]
print("data shape :" ,data.shape)
print("labels shape :",labels.shape)

