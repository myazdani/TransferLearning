import pandas as pd
import cv2
from pylab import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
import itertools

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
y_train =  list(df_train.which_of_these_sentiment_scores_does_the_above_image_fit_into_best)
print "num examples from X", len(X_train)
print "num examples and labels from y", len(y_train)

layers = [l[0] for l in DLmodel.layer_sizes]
layers.remove('data')
layers = ['norm1', 'norm2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
layers = ['prob', 'fc8', 'fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'norm2', 'norm1']

best_params = []
best_scores = []
layer_names = []

for layer_pair in itertools.combinations(layers, 2):
    layer_1 = layer_pair[0]
    layer_2 = layer_pair[1]
    print '********************************************************************'
    print 'working on layer pair', layer_pair
    print '********************************************************************'
    layer_1_pipe = Pipeline([('ANN', SkiCaffe(layer_name = layer_1, caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained))])
    layer_2_pipe = Pipeline([('ANN', SkiCaffe(layer_name = layer_2, caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained))])
    layer_union = FeatureUnion([('layer_1', layer_1_pipe), ('layer_2', layer_2_pipe)])
    #feature_pipe = Pipeline([('ANN', layer_union),('pca', RandomizedPCA(n_components=1000, whiten = False))])
    feature_pipe = Pipeline([('ANN', layer_union)])
    feaures = feature_pipe.fit_transform(X_train)
    print '********************************************************************'
    print 'feaures shape is', feaures.shape
    print 'classifying', layer_pair
    print '********************************************************************'

    lr_pipe = Pipeline([('lr', LogisticRegression(solver = 'lbfgs', multi_class = "multinomial", max_iter=5000))])
    param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'lr__C': param_range}]
    gs = GridSearchCV(estimator = lr_pipe, param_grid = param_grid, scoring = 'log_loss', cv = 5, n_jobs=-1)
    gs.fit(feaures, LabelEncoder().fit_transform(y_train))
    layer_names.append(layer_pair)
    best_params.append(gs.best_params_)
    best_scores.append(gs.best_score_)
    temp_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
    temp_df.to_pickle("../../results/CrowdFlower-LR" + layer_1 + "_" + layer_2 + ".pkl")


print '*************************************************'
print 'classifications complete'
print '*************************************************'
results_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
results_df.head()

results_df.to_pickle("../../results/CrowdFlower-LR-layer-pairs.pkl")
