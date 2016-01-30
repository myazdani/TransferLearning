
import pandas as pd
import cv2
from pylab import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

df_train = pd.read_csv("../../data/CrowdFlowerImageSentiment/sentiment-train.csv")
df_test = pd.read_csv("../../data/CrowdFlowerImageSentiment/sentiment-test.csv")

image_paths_train = list(df_train["image_paths"])
image_paths_test = list(df_test["image_paths"])


# ## Setup Trained Network

from util.skicaffe import SkiCaffe

caffe_root = '/usr/local/src/caffe/caffe-master/'
model_prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
DLmodel = SkiCaffe(caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained, layer_name = 'fc8')
DLmodel.fit()
print 'Number of layers:', len(DLmodel.layer_sizes)
DLmodel.layer_sizes


# In[42]:

X_train = image_paths_train
layers = ['prob', 'fc8', 'fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'norm2', 'norm1']


for layer in layers:
    print '********************************************************************'
    print 'working on layer', layer
    print '********************************************************************'
    feature_pipe = Pipeline([('ANN', SkiCaffe(caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained, layer_name = layer))])
    features = feature_pipe.fit_transform(X_train)
    df_train["features"] = features.tolist()
    df_train.to_pickle("../../data/CrowdFlowerImageSentiment/features/"+layer+"_features.pkl")

