
import pickle
from collections import defaultdict
from csv import DictReader


def build_id_lkp():
    user_event_clusters = defaultdict(set)
    test_id_lkp = defaultdict(str)
    for i, row in enumerate(DictReader(open("data/train.csv"))):
        if i%1000000 == 0:
            print i
        user_event_key = (row["user_location_country"], 
                          row["user_location_region"], 
                          row["user_location_city"],
                          row["hotel_market"],
                          row["orig_destination_distance"])
        user_event_clusters[user_event_key].add(row["hotel_cluster"])

    for i, row in enumerate(DictReader(open("data/test.csv"))):
        if i%1000000 == 0:
            print i
        user_event_key = (row["user_location_country"], 
                          row["user_location_region"], 
                          row["user_location_city"],
                          row["hotel_market"],
                          row["orig_destination_distance"])
        if user_event_key in user_event_clusters:
            pred_str = " ".join(list(user_event_clusters[user_event_key])[:5])
            if pred_str:
                test_id_lkp[i] = pred_str
    print len(test_id_lkp)
    return test_id_lkp

test_id_lkp = build_id_lkp()
with open('test_id_lkp.pkl', 'w') as f:
    pickle.dump(test_id_lkp, f)
