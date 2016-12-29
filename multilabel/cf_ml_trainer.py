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

X_train = np.array(["1000 meter row   Thruster 45 lbs (50 reps)   Pull-ups (30 reps)",
                    "Deadlift 1   Bench BW  Clean 3/4 BW",
                    "95 pound Squat clean, 30 reps  30 Pull-ups   Run 800 meters",
                    "40 pound Dumbbell snatch, 21 reps, right arm   21 L Pull-ups   40 pound Dumbbell snatch, 21 reps, left arm   21 L Pull-ups",
                    "2 Muscle-ups   4 Handstand Push-ups   8 2-Pood Kettlebell swings",
                    "9 Muscle-ups   15 Burpee pull-ups    21 Pull-ups   Run 800 meters",
                    "Run 800 meters  80 Squats   8 Muscle-ups",
                    "20 Pull-ups   30 Push-ups   40 Sit-ups   50 Squats",
                    "5 Pull-ups  10 Push-ups   15 Squats",
                    "Deadlift 225 lbs  Handstand push-ups",
                    "Clean 135 lbs  Ring Dips",
                    "Thruster 95 lbs  Pull-ups ",
                    "Clean and Jerk 135 lbs",
                    "40 pound Dumbbells split clean, 15 reps   21 Pull-ups",
                    "Row 1000 Meters",
                    "55 deadlifts  55 wall-ball shots  55-calorie row  55 handstand push-ups",
                    "Front squat",
                    "Run 1 mile",
                    "21 Turkish get-ups, Right arm   50 Swings  21 Overhead squats, Left arm   50 Swings  21 Overhead squats, Right arm  50 Swings  21 Turkish get-ups, Left arm",
                    "1 mile Run  100 Pull-ups   200 Push-ups   300 Squats   1 mile Run",
                    "Bodyweight bench press (e.g., same amount on bar as you weigh)   pullups",
                    "30 reps, 2 pood Kettlebell swing 30 Burpees   30 Glute-ham sit-ups",
                    "275 pound Deadlift, 5 reps  13 Push-ups  9 Box jumps, 24 inch box",
                    "115 pound Push press, 10 reps   10 KB Swings, 1.5 pood  10 Box jumps, 24 inch box",
                    "Box jumps",
                    "kettlebell 100 swings",
                    "100 pound squat",
                    "20 Turkish getups",
                    "air squat",
                    "50 pound squat",
                     "100 squat box jump 100 pull up",
                    "135-lb. thrusters, 10 reps double-unders"
                    ])
exercise_train_text = [
                ["lifting", "gymnastics"],["lifting"],["lifting", "gymnastics" , "calisthenics"],
                ["lifting","calisthenics"],["calisthenics", "kettlebell"], ["calisthenics"], 
                ["calisthenics"],["calisthenics"],["calisthenics"],
                ["lifting, calisthenics"], ["lifting", "gymnastics"],["lifting", "calisthenics"],
                ["lifting"],["lifting","calisthenics"],["calisthenics"],
                ["kettlebell"],["calisthenics"],
                ["calisthenics"],["kettlebell, calisthenics"],["lifting, calisthenics, plyometrics"],
                ["calisthenics"],["kettlebell"],["lifting, plyometrics,calisthenics"],
                ["lifting, kettlebell, plyometrics"],["plyometrics"], ["kettlebell"],
                ["lifting"],["kettlebell"],["calisthenics"],
                ["lifting"], ["lifting", "calisthenics", "gymnastics"],
                ["lifting","calisthenics","plyometrics"]
                ]

skills_train_text = [
                [""],[""],["", ""],
                ["",""],["", ""], [""], 
                [""],[""],[""],
                [", "], ["", ""],["", ""],
                [""],["",""],[""],
                [""],[""],
                [""],[", "],[", , "],
                [""],[""],[", ,"],
                [", , "],[""], [""],
                [""],[""],[""],
                [""],[""],[""]
                ]
                
if (X_train.size != len(exercise_train_text)):
    print "Training sample size %d, Exercise Labels %d Do not match" % ( X_train.size, len(exercise_train_text))
    sys.exit()    

if (X_train.size != len(skills_train_text)):
    print "Training sample size %d, Skills Labels %d Do not match" % ( X_train.size, len(skills_train_text))
    sys.exit()
    
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(exercise_train_text)

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
#predicted = classifier.predict(X_test)
#all_labels = lb.inverse_transform(predicted)

#for item, labels in zip(X_test, all_labels):
#    print '%s => %s' % (item, ', '.join(labels))
    
# save the classifier
_ = joblib.dump(ex_classifier, 'cf_ml_exercise.pkl', compress=9)
_ = joblib.dump(lb, 'cf_ml_binerizer.pkl', compress=9)

print "Successfully generated the classfier and binarizer"