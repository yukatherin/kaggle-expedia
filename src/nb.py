from __future__ import division
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse, io

import ml_metrics as metrics

# Read data
train = pd.read_csv('data2/train_booking_1.csv')
test = pd.read_csv('data2/train_booking_2.csv')
# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')


def featurize(df, cv, is_train):
    # df['has_children'] = map(lambda x: 'haschildren_'+str(x), df.srch_children_cnt > 0)
    # df['is_single'] = map(lambda x:'single_'+str(x), (df.srch_adults_cnt==1) & (df.srch_children_cnt==0))
    # df['is_couple'] = map(lambda x:'couple_'+str(x), (df.srch_adults_cnt==2) & (df.srch_children_cnt==0))
    df['main_feat'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_country, df.srch_destination_id, df.hotel_continent, df.hotel_country))
    df['main_feat2'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_country, df.srch_destination_id, df.hotel_continent, df.hotel_country, df.hotel_market))
    df['main_feat3'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_region, df.srch_destination_id, df.hotel_continent, df.hotel_country))
    df['main_feat4'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_city, df.srch_destination_id, df.hotel_continent, df.hotel_country))
    df['main_feat5'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_id, df.srch_destination_id))
    df['main_feat6'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_id, df.hotel_market))
    df['tokenized'] = map(lambda x:' '.join(map(str,x)), zip(df.main_feat, df.main_feat2, df.hotel_market, df.main_feat5, df.user_id, df.main_feat6))
    if is_train:
        feat = cv.fit_transform(df['tokenized'])
    else:
        feat = cv.transform(df['tokenized'])
    
    return feat


cv = CountVectorizer(max_features=10000000)

# Train
Xtrain = featurize(train, cv, is_train=True)
Ytrain = train.hotel_cluster

# Test
Xtest = featurize(test, cv, is_train=False)
try:
    Ytest = test.hotel_cluster
except:
    pass

io.savemat('data2/booking12dump.mat', dict(Xtrain=Xtrain.astype('double'), Ytrain=scipy.sparse.csr_matrix(Ytrain).astype('double'), Xtest=Xtest.astype('double')))




# NB w sample_weight
clf = MultinomialNB(alpha = 0.07)
clf.fit(Xtrain, Ytrain) #sample_weight = 0.1 + 0.5*train.is_booking)

pred = clf.predict_proba(Xtest)
pred_rank = np.apply_along_axis(lambda x: np.argsort(-x)[:5], 1, pred)
print pred_rank.shape
# pred_rank_prob = np.apply_along_axis(lambda x: x[np.argsort(-x)[:4]], 1, pred)

# compute_map
if Ytest.shape[0]==pred_rank.shape[0]:
    map_pred = metrics.mapk([[l] for l in Ytest], pred_rank, k=5)
    print map_pred


# pred = clf.predict(Xtest)
# acc =  sum(pred==Ytest)/len(Ytest)
# print acc

# write output
import pickle
with open('test_id_lkp.pkl') as f:
    test_id_lkp = pickle.load(f)
print len(test_id_lkp)

with open('featurized/Xtest_train_test_users_click_10.pkl', 'w') as f:
    pickle.dump(Xtest, f)

lkp_ct = 0
with open('submissions/nb_submission2.csv', 'w') as f:
    f.write("id,hotel_cluster\n")
    for i,row in enumerate(pred_rank):
        if i%1000000==0:
            print i
        if i in test_id_lkp:
            lkp_ct += 1
            f.write("%d,%s\n"%(i, test_id_lkp[i]))
            # print test_id_lkp[i], pred_rank[i]
        else:
            f.write("%d,%s\n"%(i, ' '.join(map(str,pred_rank[i]))))
print lkp_ct







# def featurize(df, cv, is_train):
#     # df['has_children'] = map(lambda x: 'haschildren_'+str(x), df.srch_children_cnt > 0)
#     # df['is_single'] = map(lambda x:'single_'+str(x), (df.srch_adults_cnt==1) & (df.srch_children_cnt==0))
#     # df['is_couple'] = map(lambda x:'couple_'+str(x), (df.srch_adults_cnt==2) & (df.srch_children_cnt==0))
#     df['main_feat'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_country, df.srch_destination_id, df.hotel_continent, df.hotel_country))
#     df['main_feat2'] = map(lambda x:str(x).replace(", ", "x"), zip(df.user_location_country, df.srch_destination_id, df.hotel_continent, df.hotel_country, df.hotel_market))
#     df['tokenized'] = map(lambda x:' '.join(map(str,x)), zip(df.main_feat, df.main_feat2, df.hotel_market))

#     if is_train:
#         feat = cv.fit_transform(df['tokenized'])
#     else:
#         feat = cv.transform(df['tokenized'])
    
#     return feat


# cv = CountVectorizer(max_features=5000000)

# # Train
# train = pd.read_csv('data2/train_booking_1.csv')
# test = pd.read_csv('data2/train_booking_2.csv')

# Xtrain = featurize(train, cv, is_train=True)
# Ytrain = train.hotel_cluster

# # NB
# clf = MultinomialNB(alpha = 0.1)
# clf.fit(Xtrain, Ytrain)

# # Test
# Xtest = featurize(test, cv, is_train=False)
# Ytest = test.hotel_cluster

# pred = clf.predict(Xtest)
# acc =  sum(pred==Ytest)/len(Ytest)
# print acc #0.1836

Xtrain = Xtrain(1:1713200,:);
Ytrain = Ytrain(1:1713200);
ytrain = np.equal(np.tile(Ytrain.T,[1,100]), np.tile(range(0,100), [Ytrain.shape[0],1]))


nn = nnsetup([784 100 10]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, Xtrain, Ytrain, opts);


