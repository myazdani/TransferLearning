import sys
import os
sys.path.insert(0, "../SkiCaffe/")
from skicaffe import SkiCaffe

caffe_root = '/usr/local/src/caffe/caffe-master/'
DLmodel = SkiCaffe('/usr/local/src/caffe/caffe-master/')
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

DLmodel.fit(model_prototxt_path = model_prototxt, model_trained_path = model_trained)


src_path, image_type = "../data/CUB/", ".jpg"
image_paths = []  
for root, dirs, files in os.walk(src_path):
    image_paths.extend([os.path.join(root, f) for f in files if f.endswith(image_type)])

for key in DLmodel.layer_dict.keys():
  if key != 'pool5/7x7_s1':
    continue
  image_features = DLmodel.transform(image_paths = image_paths, layer_name = key, return_type = 'pandasDF')
  image_features.to_csv("../features/GoogLeNet_out.csv")
