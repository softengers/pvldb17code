'''
C:\Software\Anaconda3\python.exe F:/_academic/ongoing(aca)/s1#pvldb2017/实验代码/01=USG/recommendation.py
train_data size is = 566791
training_matrix sizes are = 18737
Training User-based Collaborative Filtering...
Done. Elapsed time: 465.08991599082947 s
Precomputing similarity between friends...
Done. Elapsed time: 44.601948261260986 s
Fitting distance distribution...
Done. Elapsed time: 108.20498728752136 s
0 5151 pre@10: 0.0 rec@10: 0.0
1 6816 pre@10: 0.0 rec@10: 0.0
2 16208 pre@10: 0.06666666666666667 rec@10: 0.13333333333333333
3 2368 pre@10: 0.05 rec@10: 0.1
4 15618 pre@10: 0.04 rec@10: 0.08
5 12844 pre@10: 0.05000000000000001 rec@10: 0.10000000000000002
6 16694 pre@10: 0.042857142857142864 rec@10: 0.08571428571428573
7 5402 pre@10: 0.037500000000000006 rec@10: 0.07500000000000001
8 6686 pre@10: 0.044444444444444446 rec@10: 0.0851851851851852
9 12799 pre@10: 0.06000000000000001 rec@10: 0.09666666666666668
10 7999 pre@10: 0.06363636363636364 rec@10: 0.11060606060606061
11 4034 pre@10: 0.07500000000000001 rec@10: 0.1125
12 11578 pre@10: 0.06923076923076923 rec@10: 0.10384615384615385
13 14381 pre@10: 0.08571428571428573 rec@10: 0.15
14 2901 pre@10: 0.08000000000000002 rec@10: 0.14
15 7761 pre@10: 0.07500000000000001 rec@10: 0.13124999999999998
16 6481 pre@10: 0.07058823529411766 rec@10: 0.12352941176470586
17 10827 pre@10: 0.06666666666666668 rec@10: 0.11666666666666664
18 8290 pre@10: 0.06315789473684212 rec@10: 0.11052631578947367
19 3085 pre@10: 0.06000000000000001 rec@10: 0.10499999999999998
20 11673 pre@10: 0.05714285714285715 rec@10: 0.09999999999999998
21 4191 pre@10: 0.059090909090909104 rec@10: 0.10303030303030301

2879 13170 pre@10: 0.04927083333333333 rec@10: 0.06954442231728364

Process finished with exit code -1（2879，5hours)
9741 5956 pre@10: 0.048870868404845 rec@10: 0.06845543410076953（5hours)

'''
import numpy as np
from collections import defaultdict

from lib.UserBasedCF import UserBasedCF
from lib.FriendBasedCF import FriendBasedCF
from lib.PowerLaw import PowerLaw

from lib.metrics import precisionk, recallk


def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_relations = defaultdict(list)
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_relations[uid1].append(uid2)
        social_relations[uid2].append(uid1)
    return social_relations


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos


def read_training_data():

    train_data = open(train_file, 'r').readlines()
    #打印train_data大小，需要大内存
    print("train_data size is =", len(train_data))

    #初始化全零
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        training_matrix[uid, lid] = 1.0

    #打印training_matrix 大小，需要大内存
    print("training_matrix sizes are =", len(training_matrix))
    #print(training_matrix.shape())

    return training_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


def main():
    training_matrix = read_training_data()
    social_relations = read_friend_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    U.pre_compute_rec_scores(training_matrix)
    # U.load_result("./tmp/")

    S.compute_friend_sim(social_relations, training_matrix)
    G.fit_distance_distribution(training_matrix, poi_coos)

    result_out = open("./result/sigir11_top_" + str(top_k) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    precision, recall = [], []
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            U_scores = normalize([U.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            S_scores = normalize([S.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            G_scores = normalize([G.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])

            U_scores = np.array(U_scores)
            S_scores = np.array(S_scores)
            G_scores = np.array(G_scores)

            overall_scores = (1.0 - alpha - beta) * U_scores + alpha * S_scores + beta * G_scores

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            precision.append(precisionk(actual, predicted[:10]))
            recall.append(recallk(actual, predicted[:10]))

            print(cnt, uid, "pre@10:", np.mean(precision), "rec@10:", np.mean(recall))
            result_out.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')


if __name__ == '__main__':
    data_dir = "../data/"

    size_file = data_dir + "Gowalla_data_size.txt"
    check_in_file = data_dir + "Gowalla_checkins.txt"
    train_file = data_dir + "Gowalla_train.txt"
    tune_file = data_dir + "Gowalla_tune.txt"
    test_file = data_dir + "Gowalla_test.txt"
    social_file = data_dir + "Gowalla_social_relations.txt"
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100
    alpha = 0.1
    beta = 0.1

    U = UserBasedCF()
    S = FriendBasedCF(eta=0.05)
    G = PowerLaw()

    main()
