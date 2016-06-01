from __future__ import division
from csv import DictReader
from collections import defaultdict
from datetime import datetime
import numpy as np


start = datetime.now()

def get_top5(d):
    return " ".join(sorted(d, key=d.get, reverse=True)[:5])

def get_people_type(row):
  if int(row['srch_adults_cnt'])==1 and int(row['srch_children_cnt'])==0:
    return 1
  if int(row['srch_adults_cnt'])==2 and int(row['srch_children_cnt'])==0:
    return 2
  else:
    return 3
def score_map_at5(pred_str, hotel_cluster):
  num = 0
  map_r = list()
  for i, h in enumerate(pred_str.split()):
    if h==hotel_cluster:
      num += 1
      return (num/float(i+1))
  return 0.0

bk_destination_clusters = defaultdict(lambda: defaultdict(int))
bk_destination_pkg_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
bk_hotel_market_clusters = defaultdict(lambda: defaultdict(int))
bk_destination_couple_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
bk_destination_market_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
ck_destination_clusters = defaultdict(lambda: defaultdict(int))
ck_destination_pkg_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
ck_hotel_market_clusters = defaultdict(lambda: defaultdict(int))
ck_destination_couple_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
ck_destination_market_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

user_event_clusters = defaultdict(set)
for i, row in enumerate(DictReader(open("../data/train.csv"))):
    if (int(row['is_booking'])):
      bk_destination_clusters[row["srch_destination_id"]][row["hotel_cluster"]] += 1
      bk_destination_pkg_clusters[row['srch_destination_id']][row['is_package']][row['hotel_cluster']] += 1
      bk_hotel_market_clusters[row['srch_destination_id']][row['hotel_cluster']] += 1
      bk_destination_market_clusters[row['srch_destination_id']][row['hotel_market']][row['hotel_cluster']] += 1
      bk_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)][row['hotel_cluster']] += 1
    else:
      ck_destination_clusters[row["srch_destination_id"]][row["hotel_cluster"]] += 1
      ck_destination_pkg_clusters[row['srch_destination_id']][row['is_package']][row['hotel_cluster']] += 1
      ck_hotel_market_clusters[row['srch_destination_id']][row['hotel_cluster']] += 1
      ck_destination_market_clusters[row['srch_destination_id']][row['hotel_market']][row['hotel_cluster']] += 1
      ck_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)][row['hotel_cluster']] += 1  
    user_event_key = (row["user_location_country"], 
                      row["user_location_region"], 
                      row["user_location_city"],
                      row["hotel_market"],
                      row["orig_destination_distance"])
    user_event_clusters[user_event_key].add(row["hotel_cluster"])
    if i % 1000000 == 0:
        print("%s\t%s"%(i, datetime.now() - start))


HotelMarketFactor = 1.0
BothFactor = 5.0
PeopleFactor = 12.0
BookingFactor = 5.0
PackageFactor = 15.0
# validate
map_at5 = list()
for i, row in enumerate(DictReader(open("../data/train_booking_2.csv"))):
    to_score = defaultdict(int)
    bk_market_dict = bk_hotel_market_clusters[row['hotel_market']]
    bk_dest_dict = bk_destination_clusters[row['srch_destination_id']]
    bk_dest_pkg_dict = bk_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
    bk_dest_hotel_market_dict = bk_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
    bk_dest_ppl_dict = bk_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]
    ck_market_dict = ck_hotel_market_clusters[row['hotel_market']]
    ck_dest_dict = ck_destination_clusters[row['srch_destination_id']]
    ck_dest_pkg_dict = ck_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
    ck_dest_hotel_market_dict = ck_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
    ck_dest_ppl_dict = ck_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]    
    for h in bk_market_dict:
        to_score[h] += HotelMarketFactor*BookingFactor*np.log(bk_market_dict[h]+1.1)
    for h in bk_dest_ppl_dict:
        to_score[h] += PeopleFactor     *BookingFactor*np.log(bk_dest_ppl_dict[h]+1.1)
    for h in bk_dest_hotel_market_dict:
        to_score[h] += BothFactor       *BookingFactor*np.log(bk_dest_hotel_market_dict[h]+1.1)
    for h in bk_dest_dict:
        to_score[h] +=                   BookingFactor*np.log(bk_dest_dict[h]+1.1)
    for h in bk_dest_pkg_dict:
        to_score[h] += PackageFactor    *BookingFactor*np.log(bk_dest_pkg_dict[h]+1.1)
    for h in ck_market_dict:
        to_score[h] += HotelMarketFactor *np.log(ck_market_dict[h]+1.1)
    for h in ck_dest_ppl_dict:
        to_score[h] += PeopleFactor      *np.log(ck_dest_ppl_dict[h]+1.1)
    for h in ck_dest_hotel_market_dict:
        to_score[h] += BothFactor        *np.log(ck_dest_hotel_market_dict[h]+1.1)
    for h in ck_dest_dict:
        to_score[h] += BookingFactor     *np.log(ck_dest_dict[h]+1.1)
    for h in ck_dest_pkg_dict:
        to_score[h] += PackageFactor     *np.log(ck_dest_pkg_dict[h]+1.1)
    predstr = get_top5(to_score)
    map_at5.append(score_map_at5(predstr, row['hotel_cluster']))
    if i % 1000 == 0:
        print("%s\t%s"%(i, datetime.now() - start))
        if i>0:
          print np.mean(map_at5)
        if i==200000:
          break
print 'map at 5', np.mean(map_at5)




with open("../submissions/most_frequent_booking_7.csv", "w") as outfile:
    outfile.write("id,hotel_cluster\n")
    for i, row in enumerate(DictReader(open("../data/test.csv"))):
        user_event_key = (row["user_location_country"], 
                          row["user_location_region"], 
                          row["user_location_city"],
                          row["hotel_market"],
                          row["orig_destination_distance"])
        if user_event_key in user_event_clusters:
            pred_str = " ".join(list(user_event_clusters[user_event_key])[:5])
            outfile.write("%d,%s\n"%(i, pred_str))
        else:
            to_score = defaultdict(int)
            bk_market_dict = bk_hotel_market_clusters[row['hotel_market']]
            bk_dest_dict = bk_destination_clusters[row['srch_destination_id']]
            bk_dest_pkg_dict = bk_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
            bk_dest_hotel_market_dict = bk_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
            bk_dest_ppl_dict = bk_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]
            ck_market_dict = ck_hotel_market_clusters[row['hotel_market']]
            ck_dest_dict = ck_destination_clusters[row['srch_destination_id']]
            ck_dest_pkg_dict = ck_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
            ck_dest_hotel_market_dict = ck_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
            ck_dest_ppl_dict = ck_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]    
            for h in bk_market_dict:
                to_score[h] += HotelMarketFactor*BookingFactor*np.log(bk_market_dict[h]+1.1)
            for h in bk_dest_ppl_dict:
                to_score[h] += PeopleFactor     *BookingFactor*np.log(bk_dest_ppl_dict[h]+1.1)
            for h in bk_dest_hotel_market_dict:
                to_score[h] += BothFactor       *BookingFactor*np.log(bk_dest_hotel_market_dict[h]+1.1)
            for h in bk_dest_dict:
                to_score[h] +=                   BookingFactor*np.log(bk_dest_dict[h]+1.1)
            for h in bk_dest_pkg_dict:
                to_score[h] += PackageFactor    *BookingFactor*np.log(bk_dest_pkg_dict[h]+1.1)
            for h in ck_market_dict:
                to_score[h] += HotelMarketFactor *np.log(ck_market_dict[h]+1.1)
            for h in ck_dest_ppl_dict:
                to_score[h] += PeopleFactor      *np.log(ck_dest_ppl_dict[h]+1.1)
            for h in ck_dest_hotel_market_dict:
                to_score[h] += BothFactor        *np.log(ck_dest_hotel_market_dict[h]+1.1)
            for h in ck_dest_dict:
                to_score[h] += BookingFactor     *np.log(ck_dest_dict[h]+1.1)
            for h in ck_dest_pkg_dict:
                to_score[h] += PackageFactor     *np.log(ck_dest_pkg_dict[h]+1.1)
            predstr = get_top5(to_score)
            outfile.write("%d,%s\n"%(i,predstr))
        if i % 100000 == 0:
            print("%s\t%s"%(i, datetime.now() - start))

with open("../submissions/most_frequent_booking_9.csv", "w") as outfile:
    outfile.write("id,hotel_cluster\n")
    for i, row in enumerate(DictReader(open("../submissions/most_frequent_booking_7.csv"))):
        if not row['hotel_cluster']:
            print 'filling most popular'
            row['hotel_cluster'] = '91 48 42 59 28'
        outfile.write("%d,%s\n"%(int(row['id']), row['hotel_cluster']))


with open("../submissions/predicted_with_pandas_test.csv", "w") as outfile:
    outfile.write("id,hotel_cluster\n")
    for i, row in enumerate(DictReader(open("../submissions/predicted_with_pandas.csv"))):
        if int(row['id']) in id_pred_lkp:
            pred_str = id_pred_lkp['id']
            outfile.write("%d,%s\n"%(i, pred_str))
        else:
            outfile.write("%d,%s\n"%(int(row['id']), row['hotel_cluster']))


# HotelMarketFactor = 1.0
# BothFactor = 10.0
# PeopleFactor = 12.0
# BookingFactor = 10.0
# PackageFactor = 15.0
# # validate
# map_at5 = list()
# for i, row in enumerate(DictReader(open("../data/train_booking_2.csv"))):
#     to_score = defaultdict(int)
#     bk_market_dict = bk_hotel_market_clusters[row['hotel_market']]
#     bk_dest_dict = bk_destination_clusters[row['srch_destination_id']]
#     bk_dest_pkg_dict = bk_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
#     bk_dest_hotel_market_dict = bk_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
#     bk_dest_ppl_dict = bk_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]
#     ck_market_dict = ck_hotel_market_clusters[row['hotel_market']]
#     ck_dest_dict = ck_destination_clusters[row['srch_destination_id']]
#     ck_dest_pkg_dict = ck_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
#     ck_dest_hotel_market_dict = ck_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
#     ck_dest_ppl_dict = ck_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]    
#     for h in bk_market_dict:
#         to_score[h] += HotelMarketFactor*BookingFactor*bk_market_dict[h]
#     for h in bk_dest_ppl_dict:
#         to_score[h] += PeopleFactor     *BookingFactor*bk_dest_ppl_dict[h]
#     for h in bk_dest_hotel_market_dict:
#         to_score[h] += BothFactor       *BookingFactor*bk_dest_hotel_market_dict[h]
#     for h in bk_dest_dict:
#         to_score[h] +=                   BookingFactor*bk_dest_dict[h]
#     for h in bk_dest_pkg_dict:
#         to_score[h] += PackageFactor    *BookingFactor*bk_dest_pkg_dict[h]
#     for h in ck_market_dict:
#         to_score[h] += HotelMarketFactor *ck_market_dict[h]
#     for h in ck_dest_ppl_dict:
#         to_score[h] += PeopleFactor      *ck_dest_ppl_dict[h]
#     for h in ck_dest_hotel_market_dict:
#         to_score[h] += BothFactor        *ck_dest_hotel_market_dict[h]
#     for h in ck_dest_dict:
#         to_score[h] += BookingFactor     *ck_dest_dict[h]
#     for h in ck_dest_pkg_dict:
#         to_score[h] += PackageFactor     *ck_dest_pkg_dict[h]
#     predstr = get_top5(to_score)
#     map_at5.append(score_map_at5(predstr, row['hotel_cluster']))
#     if i % 1000 == 0:
#         print("%s\t%s"%(i, datetime.now() - start))
#         if i>0:
#           print np.mean(map_at5)
#         if i==200000:
#           break
# print 'map at 5', np.mean(map_at5)


id_pred_lkp = defaultdict(str)

with open("../submissions/most_frequent_booking_null.csv", "w") as outfile:
    outfile.write("id,hotel_cluster\n")
    for i, row in enumerate(DictReader(open("../data/test.csv"))):
        user_event_key = (row["user_location_country"], 
                          row["user_location_region"], 
                          row["user_location_city"],
                          row["hotel_market"],
                          row["orig_destination_distance"])
        if user_event_key in user_event_clusters:
            pred_str = " ".join(list(user_event_clusters[user_event_key])[:5])
            id_pred_lkp[i] = pred_str
            outfile.write("%d,%s\n"%(i, pred_str))
        else:
            pass
            # to_score = defaultdict(int)
            # bk_market_dict = bk_hotel_market_clusters[row['hotel_market']]
            # bk_dest_dict = bk_destination_clusters[row['srch_destination_id']]
            # bk_dest_pkg_dict = bk_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
            # bk_dest_hotel_market_dict = bk_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
            # bk_dest_ppl_dict = bk_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]
            # ck_market_dict = ck_hotel_market_clusters[row['hotel_market']]
            # ck_dest_dict = ck_destination_clusters[row['srch_destination_id']]
            # ck_dest_pkg_dict = ck_destination_pkg_clusters[row['srch_destination_id']][row['is_package']]
            # ck_dest_hotel_market_dict = ck_destination_market_clusters[row['srch_destination_id']][row['hotel_market']]
            # ck_dest_ppl_dict = ck_destination_couple_clusters[row['srch_destination_id']][get_people_type(row)]    
            # for h in bk_market_dict:
            #     to_score[h] += HotelMarketFactor*BookingFactor*bk_market_dict[h]
            # for h in bk_dest_ppl_dict:
            #     to_score[h] += PeopleFactor     *BookingFactor*bk_dest_ppl_dict[h]
            # for h in bk_dest_hotel_market_dict:
            #     to_score[h] += BothFactor       *BookingFactor*bk_dest_hotel_market_dict[h]
            # for h in bk_dest_dict:
            #     to_score[h] +=                   BookingFactor*bk_dest_dict[h]
            # for h in bk_dest_pkg_dict:
            #     to_score[h] += PackageFactor    *BookingFactor*bk_dest_pkg_dict[h]
            # for h in ck_market_dict:
            #     to_score[h] += HotelMarketFactor *ck_market_dict[h]
            # for h in ck_dest_ppl_dict:
            #     to_score[h] += PeopleFactor      *ck_dest_ppl_dict[h]
            # for h in ck_dest_hotel_market_dict:
            #     to_score[h] += BothFactor        *ck_dest_hotel_market_dict[h]
            # for h in ck_dest_dict:
            #     to_score[h] += BookingFactor     *ck_dest_dict[h]
            # for h in ck_dest_pkg_dict:
            #     to_score[h] += PackageFactor     *ck_dest_pkg_dict[h]
            # predstr = get_top5(to_score)
            # outfile.write("%d,%s\n"%(i,predstr))
        if i % 100000 == 0:
            print("%s\t%s"%(i, datetime.now() - start))






