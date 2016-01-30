import pandas as pd
import cv2
from pylab import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize, LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
import itertools
import os
import gc

src_path, file_type = "../../data/CrowdFlowerImageSentiment/features/", ".pkl"

data_frames = []
for root, dirs, files in os.walk(src_path):
    data_frames.extend([os.path.join(root, f) for f in files if f.endswith(file_type)])

data_frames =[src_path+"prob_features.pkl", src_path+"fc8_features.pkl", src_path+"fc7_features.pkl", src_path+"fc6_features.pkl"]

best_params = []
best_scores = []
layer_names = []

def get_layer_name(a_path):
    return a_path.split("/")[-1].split("_")[0]

for df_pair in itertools.combinations(data_frames, 2):
    layer_1 = get_layer_name(df_pair[0])
    layer_2 = get_layer_name(df_pair[1])
    layer_pair = (layer_1, layer_2)
    print '********************************************************************'
    print 'working on pair', df_pair
    print '********************************************************************'
    df_1 = pd.read_pickle(df_pair[0])
    df_2 = pd.read_pickle(df_pair[1])
    features_1 = array(list(df_1.features))
    features_2 = array(list(df_2.features))
    X_train = hstack((features_1, features_2))
    y_train = list(df_1.which_of_these_sentiment_scores_does_the_above_image_fit_into_best)
    print '********************************************************************'
    print 'feaures shape is', X_train.shape
    print 'classifying', layer_pair
    print '********************************************************************'
    lr_pipe = Pipeline([('scaling', StandardScaler()), ('lr', LogisticRegression(solver = 'lbfgs', multi_class = "multinomial", max_iter=5000))])
    param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'lr__C': param_range}]
    gs = GridSearchCV(estimator = lr_pipe, param_grid = param_grid, scoring = 'accuracy', cv = 5, n_jobs=-1, pre_dispatch='3*n_jobs')
    gs.fit(X_train, LabelEncoder().fit_transform(y_train))
    layer_names.append(layer_pair)
    best_params.append(gs.best_params_)
    best_scores.append(gs.best_score_)
    temp_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
    temp_df.to_pickle("../../results/CrowdFlower-LR-scaled-" + layer_1 + "_" + layer_2 + ".pkl")
    gc.collect()


print '*************************************************'
print 'classifications complete'
print '*************************************************'
results_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
results_df.head()

results_df.to_pickle("../../results/CrowdFlower-LR-layer-pairs-scaled.pkl")
