from __future__ import division
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print 'reading data...'
df = pd.read_csv("../featurized/recall_train_booking_2.csv", sep='\t')
X = df[['srch_dest_ct', 'hotel_market_ct', 'dest_pct', 'market_pct']]
is_booked = df.pop('is_booked')

Xtrain, Xtest, ytrain, ytest = train_test_split(X, is_booked)

# print 'training rf...'
# rf = RandomForestClassifier(max_depth=3, n_estimators=100, n_jobs=-1).fit(Xtrain, ytrain)
# rf_pred = rf.predict(ytest)
# print sum(rf_pred==ytest)/len(ytest)


print 'training xgb...'
xgbm = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.01).fit(Xtrain, ytrain)
xgbm_pred = xgbm.predict(Xtest)
print sum(xgbm_pred==ytest)/len(ytest)

with open('../models/xgbm1.pkl', 'w') as f:
    pickle.dump(xgbm, f)