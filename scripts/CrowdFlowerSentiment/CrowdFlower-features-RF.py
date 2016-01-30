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
import os
import gc

src_path, file_type = "../../data/CrowdFlowerImageSentiment/features/", ".pkl"

data_frames = []
for root, dirs, files in os.walk(src_path):
    data_frames.extend([os.path.join(root, f) for f in files if f.endswith(file_type)])


best_params = []
best_scores = []
layer_names = []

def get_layer_name(a_path):
    return a_path.split("/")[-1].split("_")[0]

for df_path in data_frames:
    layer = get_layer_name(df_path)
    print '********************************************************************'
    print 'working on layer', layer
    print '********************************************************************'
    df = pd.read_pickle(df_path)
    features = array(list(df.features))
    X_train = RandomizedPCA(n_components=1000, whiten = False).fit_transform(features)
    y_train = list(df.which_of_these_sentiment_scores_does_the_above_image_fit_into_best)
    print '********************************************************************'
    print 'feaures shape is', X_train.shape
    print 'classifying', layer
    print '********************************************************************'
    lr_pipe = Pipeline([('lr', LogisticRegression(solver = 'lbfgs', multi_class = "multinomial", max_iter=5000))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'lr__C': param_range}]
    gs = GridSearchCV(estimator = lr_pipe, param_grid = param_grid, scoring = 'log_loss', cv = 5, n_jobs=-1, error_score=0)
    gs.fit(X_train, LabelEncoder().fit_transform(y_train))
    layer_names.append(layer)
    best_params.append(gs.best_params_)
    best_scores.append(gs.best_score_)
    temp_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
    temp_df.to_pickle("../../results/CrowdFlower-LR-accuracy-score-" + layer + ".pkl")


results_df = pd.DataFrame({'layer.name': layer_names, 'best_scores': best_scores, 'best_params': best_params})
results_df.head()

results_df.to_pickle("../../results/CrowdFlower-LR-accuracy-score.pkl")
