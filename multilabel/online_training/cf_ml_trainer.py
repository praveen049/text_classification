#!/usr/bin/python

import pandas as pd
import numpy as np
import sys
import cPickle as pickler
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

lb = preprocessing.MultiLabelBinarizer()

training_data = pd.DataFrame.from_csv('exercise.csv',index_col=None)
X_train = np.array(training_data.workout)
l_train = np.array(training_data[['e1','e2','e3','e4','e5','e6']], dtype=str)

Y = lb.fit_transform(l_train)
#print l_train
#print Y

if (X_train.size != len(l_train)):
    print "Training sample size %d, Exercise Labels %d Do not match" % ( X_train.size, len(l_train))
    sys.exit()    
   

ex_classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

ex_classifier.fit(X_train, Y)
    
# save the classifier
_ = joblib.dump(ex_classifier, 'cf_ml_exercise.pkl', compress=9)
_ = joblib.dump(lb, 'cf_ml_binerizer.pkl', compress=9)

print "Successfully generated the classfier and binarizer"
