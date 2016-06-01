from csv import DictReader
from collections import defaultdict
from datetime import datetime

import pandas as pd
import pickle
test = pd.read_csv('data/test.csv')
test_user_id_set = set(test.user_id)
with open('test_user_set_pkl', 'w') as f:
    pickle.dump(test_user_id_set, f)

import random 
outfile = open("data2/train_test_users_click_10.csv", 'w')
outfile.write("date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster\n")
f = open("data/train.csv")
line = f.readline()
print line
for i, row in enumerate(DictReader(open("data/train.csv"))):
    if (i % 1000000 == 0):
        print i
    line = f.readline()
    if i==0:
        print line, row
    if int(row['user_id']) in test_user_id_set:
        if int(row['is_booking']):
            outfile.write(line)
        elif random.random() < 0.1:
            outfile.write(line)

f.close()
outfile.close()





##
start = datetime.now()

f = open("../data/train.csv")
outfile1 = open("../data2/train_1.csv", 'w')
outfile2 = open("../data2/train_2.csv", 'w')

outfile1.write("date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster\n")
outfile2.write("date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster\n")
for i, row in enumerate(DictReader(open("../data/train.csv"))):
    if (i % 1000000 == 0):
        print("%s\t%s"%(i, datetime.now() - start))
    line = f.readline()
    if int(row['is_booking']):
        if (int(row['user_id'])%7%2 == 0):
            outfile1.write(line)
        else:
            outfile2.write(line)

f.close()
outfile1.close()
outfile2.close()
