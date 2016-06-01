from __future__ import division
from csv import DictReader
from collections import defaultdict
from datetime import datetime

from numpy import log


Min_Dest_Ct_Pct = 0.2
Min_Market_Ct_Pct = 0.2


start = datetime.now()

destination_clusters = defaultdict(lambda: defaultdict(int))
hotel_market_clusters = defaultdict(lambda: defaultdict(int))
user_clusters = defaultdict(lambda: defaultdict(int))

for i, row in enumerate(DictReader(open("../data/train_booking_1.csv"))):
    destination_clusters[row["srch_destination_id"]][row["hotel_cluster"]] += 1
    hotel_market_clusters[row["hotel_market"]][row["hotel_cluster"]] += 1
    if i % 1000000 == 0:
        print("%s\t%s"%(i, datetime.now() - start))

with open("../featurized/recall_train_booking_2.csv", "w") as outfile:
    outfile.write("id\thotel_cluster\tsrch_dest_ct\thotel_market_ct\tdest_pct\tmarket_pct\tis_booked\n")
    for i, row in enumerate(DictReader(open("../data/train_booking_2.csv"))):
        target_dest_cts = destination_clusters[row["srch_destination_id"]]
        target_market_cts = hotel_market_clusters[row["hotel_market"]]
        dest_max_ct = max(target_dest_cts.values()) if target_dest_cts else 0.0
        market_max_ct = max(target_market_cts.values()) if target_market_cts else 0.0
        recall_set = set(target_dest_cts.keys()).union(set(target_market_cts.keys()))
        if len(recall_set)==0:
            print 'no recall for', i
        for hotel_cluster in recall_set:
            is_booked = 1 if (hotel_cluster==row["hotel_cluster"]) else 0
            dest_ct = target_dest_cts[hotel_cluster]
            market_ct = target_market_cts[hotel_cluster]
            dest_pct = dest_ct/dest_max_ct if dest_max_ct else 0.0
            market_pct = market_ct/market_max_ct if market_max_ct else 0.0
            if (dest_pct > Min_Dest_Ct_Pct) or (market_pct > Min_Market_Ct_Pct):
                feats = (i, hotel_cluster, dest_ct, market_ct, dest_pct, market_pct, is_booked)
                outfile.write("%d\t%s\t%d\t%d\t%.3f\t%.3f\t%d\n"%feats)

