
## All featurization
## Generates 8 pkl files: train_full, train_sm, test_full, test_sm, train_dev_full, train_dev_sm, val_full, val_sm

import pandas as pd
import numpy as np
import pickle
import datetime


def parsedate(x):
    try:
        parsed = pd.to_datetime(x)
        if parsed < pd.to_datetime('2013'):
            return np.nan
        elif str(parsed.date()) > '2017' or str(parsed.date()) < '2013' or len(str(parsed.date())) != 10:
            return np.nan
        return parsed.date()
    except:
        return np.nan


def add_feat_transforms(df):

    df['date_time_dt'] = df.date_time.map(parsedate)
    df['srch_ci_dt'] = df.srch_ci.map(parsedate)
    df['srch_co_dt'] = df.srch_co.map(parsedate)

    df['book_dt_numeric'] = (df.date_time_dt - pd.to_datetime('2012-01-01').date()) / np.timedelta64(1, 'D')
    df['planahead_window_days'] = (df.srch_ci_dt - df.date_time_dt) / np.timedelta64(1, 'D')
    df['srch_staylength'] = (df['srch_co_dt'] - df['srch_ci_dt']) / np.timedelta64(1, 'D') + 1
    df['srch_ci_dow'] = df.srch_ci_dt.map(lambda x: x.weekday() if type(x)==datetime.date else 0)
    df['srch_co_dow'] = df.srch_co_dt.map(lambda x: x.weekday() if type(x)==datetime.date else 0)
    df['srch_incl_weeknd'] = (df.srch_ci_dow + df.srch_staylength) >= 5

    df = pd.concat((df, pd.get_dummies(df['srch_ci_dow'])))
    df.fillna(df.median(), inplace=True)



# Read data
train = pd.read_csv('data/train.csv', parse_dates=['date_time', 'srch_co', 'srch_ci'])
test = pd.read_csv('data/test.csv', parse_dates=['date_time', 'srch_co', 'srch_ci'])
dest = pd.read_csv('data/destinations.csv')


# Add engineered features
add_feat_transforms(train)
add_feat_transforms(test)


# Cluster counts by srch_dest_id
counts_srch_destination_id = train.groupby('srch_destination_id').agg({'is_booking': sum, 'date_time': len})
counts_srch_destination_id_hotel_cluster = train.groupby(['srch_destination_id', 'hotel_cluster']).agg({'is_booking': sum, 'date_time': len})
counts_srch_destination_id_hotel_cluster.reset_index(inplace=True)
counts_srch_destination_id_hotel_cluster['tmp_total_ct'] = counts_srch_destination_id.ix[counts_srch_destination_id_hotel_cluster.srch_destination_id, 'date_time'].values
counts_srch_destination_id_hotel_cluster['normalized_ct'] = counts_srch_destination_id_hotel_cluster['date_time'] / counts_srch_destination_id_hotel_cluster['tmp_total_ct']

hotel_cluster_counts_by_srch_destination_id = pd.pivot_table(counts_srch_destination_id_hotel_cluster, values=['is_booking', 'date_time', 'normalized_ct'], index=['srch_destination_id'], columns=['hotel_cluster'], aggfunc=max)
hotel_cluster_counts_by_srch_destination_id.fillna(0, inplace=True)

# Cluster counts by hotel_market
counts_hotel_market = train.groupby('hotel_market').agg({'is_booking': sum, 'date_time': len})
counts_hotel_market_hotel_cluster = train.groupby(['hotel_market', 'hotel_cluster']).agg({'is_booking': sum, 'date_time': len})
counts_hotel_market_hotel_cluster.reset_index(inplace=True)
counts_hotel_market_hotel_cluster['tmp_total_ct'] = counts_hotel_market.ix[counts_hotel_market_hotel_cluster.hotel_market, 'date_time'].values
counts_hotel_market_hotel_cluster['normalized_ct'] = counts_hotel_market_hotel_cluster['date_time'] / counts_hotel_market_hotel_cluster['tmp_total_ct']

hotel_cluster_counts_by_hotel_market = pd.pivot_table(counts_hotel_market_hotel_cluster, values=['is_booking', 'date_time', 'normalized_ct'], index=['hotel_market'], columns=['hotel_cluster'], aggfunc=max)
hotel_cluster_counts_by_hotel_market.fillna(0, inplace=True)

# Cluster counts by user_id



# Join with cluster counts



# PCA
# from sklearn.decomposition import PCA
# dest.set_index('hotel_market', inplace=True)
# pca = PCA(n_components=3)
# dest_pca = pca.fit_transform(dest)
# dest_pca = pd.DataFrame(dest_pca)
# dest_pca['srch_destination_id'] = dest.index

# Join with destinations latent variables
train_dest = pd.merge(train, dest_pca, on='srch_destination_id')
test_dest = pd.merge(test, dest_pca, on='srch_destination_id')


# Validation sample


# Small


# Full






