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
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

X_train = np.array(["1000 meter row   Thruster 45 lbs (50 reps)   Pull-ups (30 reps)", #1
                    "Deadlift 1   Bench BW  Clean 3/4 BW",#2
                    "95 pound Squat clean, 30 reps  30 Pull-ups   Run 800 meters", #3
                    "40 pound Dumbbell snatch, 21 reps, right arm   21 L Pull-ups   40 pound Dumbbell snatch, 21 reps, left arm   21 L Pull-ups", #4
                    "2 Muscle-ups   4 Handstand Push-ups   8 2-Pood Kettlebell swings", #5
                    "9 Muscle-ups   15 Burpee pull-ups    21 Pull-ups   Run 800 meters", #6
                    "Run 800 meters  80 Squats   8 Muscle-ups", #7
                    "20 Pull-ups   30 Push-ups   40 Sit-ups   50 Squats", #8
                    "5 Pull-ups  10 Push-ups   15 Squats", #9
                    "Deadlift 225 lbs  Handstand push-ups", #10
                    "Clean 135 lbs  Ring Dips", #11
                    "Thruster 95 lbs  Pull-ups ", #12
                    "Clean and Jerk 135 lbs", #13
                    "40 pound Dumbbells split clean, 15 reps   21 Pull-ups", #14
                    "Row 1000 Meters", #15
                    "55 deadlifts  55 wall-ball shots  55-calorie row  55 handstand push-ups", #16
                    "Front squat", #17
                    "Run 1 mile", #18
                    "21 Turkish get-ups, Right arm   50 Swings  21 Overhead squats, Left arm   50 Swings  21 Overhead squats, Right arm  50 Swings  21 Turkish get-ups, Left arm", #19
                    "1 mile Run  100 Pull-ups   200 Push-ups   300 Squats   1 mile Run" #20

                    ])
skill_train_text = [
                ["cadio", "stamina" , "power" ,"strength","flexibility"],  #1
		["power", "strength","agility","balance"], #2
		["speed", "power" , "strength","flexibility","balance"], #3
                ["power","flexibility","strength","coordination"], #4
		["agility", "balance","strength"], #5
		["agility","speed" ,"stamina", "cardio"], #6
                ["speed","stamina", "coordination","agility"], #7
		["flexibility","coordination","stamina", "cardio"], #8
		["flexibility","coordination"], #9 
                ["strength", "power", "balance"],  #10
		["strength", "power","flexibility","agility","coordination"], #11
		["strength", "power","flexibility"], #12
                ["power","strength","accuracy"], #13
		["power","flexibility"], #14
		["power","stamina","cardio"], #15
                ["power","flexibility"], #16
		["strength", "balance"], #17
                ["cadrio","stamina"], #18
		["stamina" ,"balance","agility", "flexibility","coordination"], #19
		["cadrio", "flexibility","stamina", "power"] #20
                ]

                
if (X_train.size != len(skill_train_text)):
    print "Training sample size %d, Skills Labels %d Do not match" % ( X_train.size, len(skill_train_text))
    sys.exit()    
   
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(skill_train_text)

ex_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

ex_classifier.fit(X_train, Y)
# Test a little
#X_test = np.array(["400 meter run   Overhead squat 95 lbs x 15",
#                   "Snatch 135 pounds",
#                  "Run 1 mile",
#                   "Ring Dips",
#                   "push up, pull ups",
#                   "Box jumps",
#                   "135 pounds Squat",
#                   "turkish swing"])
#predicted = ex_classifier.predict(X_test)
#all_labels = lb.inverse_transform(predicted)

#for item, labels in zip(X_test, all_labels):
#    print '%s => %s' % (item, ', '.join(labels))
    
# save the classifier
_ = joblib.dump(ex_classifier, 'cf_ml_skills.pkl', compress=9)
_ = joblib.dump(lb, 'cf_ml_skills_binerizer.pkl', compress=9)

print "Successfully generated the classfier and binarizer"
