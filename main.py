import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

###import csv
pd.set_option('display.max_columns', None)
df_master = pd.read_csv('student-mat.csv')
print('################# ORIGINAL CSV')
print(df_master)

index = ['school',
                       'sex',
                       'age',
                       'address',
                       'famsize',

                       'Pstatus',
                       'Medu',
                       'Fedu',
                       'Mjob',
                       'Fjob',

                       'reason',
                       'guardian',
                       'traveltime',
                       'studytime',
                       'failures',

                       'schoolsup',
                       'famsup',
                       'paid',
                       'activities',
                       'nursery',

                       'higher',
                       'internet',
                       'romantic',
                       'famrel',
                       'freetime',

                       'goout',
                       'Dalc',
                       'Walc',
                       'health',
                       'absences',

                        'G1', 'G2', 'G3'
                        ]

###preprocessing/vectorizing

#binarizing some classes
df_master['G1'] = df_master['G1'].mask(df_master['G1'] < 10, 0)
df_master['G1'] = df_master['G1'].mask(df_master['G1'] >= 10, 1)

df_master['G2'] = df_master['G2'].mask(df_master['G2'] < 10, 0)
df_master['G2'] = df_master['G2'].mask(df_master['G2'] >= 10, 1)

df_master['G3'] = df_master['G3'].mask(df_master['G3'] < 10, 0)
df_master['G3'] = df_master['G3'].mask(df_master['G3'] >= 10, 1)

#encode
label_encode_these = ['school',
                      'sex',
                      'address',
                      'famsize',
                      'Pstatus',
                      'Mjob',
                      'Fjob',
                      'reason',
                      'guardian',
                      'schoolsup',
                      'famsup',
                      'paid',
                      'activities',
                      'nursery',
                      'higher',
                      'internet',
                      'romantic']
for each in label_encode_these:
    df_master[each] = df_master[each].astype('category')
    df_master.dtypes
    df_master[each] = df_master[each].cat.codes


print('################# AFTER P/NP AND BINARY STR ENCODE')
print(df_master)

###EDA
corr = df_master.corr()
corr['Gavg'] = (corr['G1'] + corr['G2'] + corr['G3']) / 3

print('corr G1')
print(corr['G1'])
corr.to_csv('corr.csv')

###data split
df_train, df_test = train_test_split(df_master, test_size= 0.2)

df_train_X = df_train[['school',
                       'sex',
                       'age',
                       'address',
                       'famsize',

                       'Pstatus',
                       'Medu',
                       'Fedu',
                       'Mjob',
                       'Fjob',

                       'reason',
                       'guardian',
                       'traveltime',
                       'studytime',
                       'failures',

                       'schoolsup',
                       'famsup',
                       'paid',
                       'activities',
                       'nursery',

                       'higher',
                       'internet',
                       'romantic',
                       'famrel',
                       'freetime',

                       'goout',
                       'Dalc',
                       'Walc',
                       'health',
                       'absences'
                        ]]
df_train_math = df_train[['G1', 'G2', 'G3']]

df_test_X = df_test[['school',
                       'sex',
                       'age',
                       'address',
                       'famsize',

                       'Pstatus',
                       'Medu',
                       'Fedu',
                       'Mjob',
                       'Fjob',

                       'reason',
                       'guardian',
                       'traveltime',
                       'studytime',
                       'failures',

                       'schoolsup',
                       'famsup',
                       'paid',
                       'activities',
                       'nursery',

                       'higher',
                       'internet',
                       'romantic',
                       'famrel',
                       'freetime',

                       'goout',
                       'Dalc',
                       'Walc',
                       'health',
                       'absences'
                        ]]
df_test_math = df_test[['G1', 'G2', 'G3']]


###training
clf = GaussianNB()
clf.fit(df_train_X, df_train_math['G1'])
print('gaussian NB acc' + str(clf.score(df_test_X, df_test_math['G1'])))
print('importance of each features')
imps = permutation_importance(clf, df_test_X, df_test_math['G1'])
imps = imps.importances_mean
print(imps)

print('conf matrix, tn, fp, fn, tp')
cm = confusion_matrix(df_test_math['G1'], clf.predict(df_test_X))
print(cm)

tn, fp, fn, tp = confusion_matrix(df_test_math['G1'], clf.predict(df_test_X)).ravel()
print (tn, fp, fn, tp)

###DO PREDICTION TEST
while 1:
    predict = []
    print('################################################ PREDICTOR')
    print("school GP=0 MS=1\n sex F=0 M=1 \n age 15-22\n address rural=0 urban=1 \n famsize <=3:1 >3 0 \n Pstat T=1 A=0"
          "\n Medu Fedu 0-4 \n Mjob Fjob athome=0 health=1 other=2 services=3 teacher=4 \n reason course=0 home=1 other=2"
          "reputation=3 \n guardian father=0 mother=1 other=2 \n traveltime 1-4 \n studytime 1-4 \n fail 0-4 \n schoolsup "
          "famsup paid activities nursery higher internet romantic no=0 yes=1 \n famrel freetime goout Dalc Walc health 1-5 "
          "\n absences 0-93")
    for i in range(0, 30, 1):
        print('input to ' + index[i])
        inp = int(input())
        predict.append(inp)

    print('prediction:' + str(clf.predict([predict])))



