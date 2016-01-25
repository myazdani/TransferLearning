import pandas as pd
import cv2
from pylab import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA

df_train = pd.read_pickle("../data/VOC2007/VOC-training.pkl")
df_test = pd.read_pickle("../data/VOC2007/VOC-testing.pkl")

image_paths_train = list(df_train["image.paths"])
image_paths_test = list(df_test["image.paths"])


from util.skicaffe import SkiCaffe

caffe_root = '/usr/local/src/caffe/caffe-master/'
model_prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
DLmodel = SkiCaffe(caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained, layer_name = 'fc8')
DLmodel.fit()
print 'Number of layers:', len(DLmodel.layer_sizes)
DLmodel.layer_sizes

X_train = image_paths_train
y_train = MultiLabelBinarizer().fit_transform(df_train.classes)
print "num examples from X", len(X_train)
print "num examples and labels from y", y_train.shape

layers = [l[0] for l in DLmodel.layer_sizes]
layers.remove('data')
layers = ['norm1', 'norm2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
layers = ['prob', 'fc8', 'fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'norm2', 'norm1']

best_params = []
best_scores = []
layer_names = []

for layer in layers:
    print '********************************************************************'
    print 'working on layer', layer
    print '********************************************************************'
    #feature_pipe = Pipeline([('ANN', SkiCaffe(caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained, layer_name = layer))])
    feature_pipe = Pipeline([('ANN', SkiCaffe(caffe_root = caffe_root,model_prototxt_path = model_prototxt, model_trained_path = model_trained, layer_name = layer)),
                            ('pca', RandomizedPCA(n_components=1000, whiten = True))])
    feaures = feature_pipe.fit_transform(X_train)
    print '********************************************************************'
    print 'feaures shape is', feaures.shape
    print 'classifying', layer
    print '********************************************************************'

    lr_pipe = Pipeline([('OVClr', OneVsRestClassifier(LogisticRegression()))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'OVClr__estimator__C': param_range}]
    gs = GridSearchCV(estimator = lr_pipe, param_grid = param_grid, scoring = 'average_precision', cv = 5, n_jobs=-1)
    gs.fit(feaures, array(y_train))
    layer_names.append(layer)
    best_params.append(gs.best_params_)
    best_scores.append(gs.best_score_)
    temp_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
    temp_df.to_pickle("../results/VOC-LR" + layer + ".pkl")


results_df = pd.DataFrame({'layer.name': layers, 'best_scores': best_scores, 'best_params': best_params})
results_df.head()

results_df.to_pickle("../results/VOC-LR.pkl")
