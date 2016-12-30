#!/usr/bin/python

import numpy as np
import sys
from sklearn.externals import joblib
import os as os
from sklearn import preprocessing


if len(sys.argv) == 1:
    print "USAGE: ./cf_ml_exer_predict.py <wod_desc>"
    sys.exit()
 
if not os.path.isfile("cf_ml_exercise.pkl"):
    print "Classfier is missing"
    sys.exit()
if not os.path.isfile("cf_ml_binerizer.pkl"):
    print "Binarizer is missing"
    sys.exit()
    
# load it again
classifier = joblib.load('cf_ml_exercise.pkl')
lb = joblib.load('cf_ml_binerizer.pkl')
X_predict = np.array(sys.argv[1:])

predicted = classifier.predict(X_predict)
all_labels = lb.inverse_transform(predicted)
print all_labels