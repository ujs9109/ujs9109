import pandas as pd
import numpy as np
import random
import itertools

# seed_number = 4
# seed_number = 40
seed_number = 1
num_FR = 3
num_option = 6
num_DP = 10

time_calculation = []
conflict_seed = []


def PFO_make(num_FR, num_option):
    PFO = np.zeros([num_FR, num_option])
    for i in range(num_FR):
        num_effected_OP = random.sample([k + 1 for k in range(num_option)], k=1)[0]

        possible_effected_OP = [k for k in range(num_option)]

        effected_OP = random.sample(possible_effected_OP, k=num_effected_OP)
        signed_OP = random.choices([1, -1], weights=(60, 40), k=num_effected_OP)

        for j in range(num_effected_OP):
            PFO[i][effected_OP[j]] = signed_OP[j]
    return PFO


def OD_make(num_option, num_DP):
    OD = np.zeros([num_option, num_DP])
    for i in range(num_option):
        num_effected_DP = random.choices([1, 2, 3], weights=(45, 45, 10), k=1)[0]

        possible_effected_DP = [k for k in range(num_DP)]

        effected_DP = random.sample(possible_effected_DP, k=num_effected_DP)
        signed_DP = random.choices([1, -1], weights=(60, 40), k=num_effected_DP)

        for j in range(num_effected_DP):
            OD[i][effected_DP[j]] = signed_DP[j]
    return OD
def one_hot_encoding(DP_list):
    fulllist = []
    for k in range(len(DP_list)):
        if DP_list[k] != 0 :
            imme_list = np.zeros(len(DP_list))
            imme_list[k] = DP_list[k]
            fulllist.append(imme_list)
    return fulllist


def checking_DP(Option_candidates):
    total_DP = []
    for i in range(num_DP):
        negative_sign = 0
        positive_sign = 0

        for j in Option_candidates:
            if np.sign(j) * OD[abs(j) - 1][i] == -1:
                negative_sign += 1
            elif np.sign(j) * OD[abs(j) - 1][i] == 1:
                positive_sign += 1

        if (negative_sign > 0) and (positive_sign > 0):
            print("conflict exists on DP-level", Option_candidates, ", DP : ", i + 1)
            return False, False

        elif (negative_sign == 0) and (positive_sign == 0):
            total_DP.append(0)
        elif negative_sign > 0:
            total_DP.append(-1)
        else:
            total_DP.append(1)

    return total_DP, Option_candidates


# 각 DP 마다 path 만들기
def path_make(DP_index):
    num_path = random.randint(1, 3)
    path_DP = np.zeros((num_path, 1, num_DP))

    for i in range(len(path_DP)):
        possible_effected_DP = [k for k in range(num_DP) if k != DP_index]
        num_effected_DP = random.choices([0, 1, 2, 3], weights=(5, 35, 50, 10), k=1)[0]
        effected_DP = random.sample(possible_effected_DP, k=num_effected_DP)
        signed_DP = random.choices([1, -1], weights=(60, 40), k=num_effected_DP)

        for j in range(num_effected_DP):
            path_DP[i][0][effected_DP[j]] = signed_DP[j]

    return path_DP


# 각 path중 하나를 골라서 path_matrix 만들기
def make_PM(combination_index):
    empty_list = []
    for i in range(len(combination_index)):
        empty_list.append(combination_index[i][0])

    return np.matrix(empty_list)

# 각 DP 마다 path 만들기
def path_make(DP_index):
    num_path = random.choices([1, 2, 3], weights=(50, 40, 10), k=1)[0]
    path_DP = np.zeros((num_path, 1, num_DP))

    for i in range(len(path_DP)):
        possible_effected_DP = [k for k in range(num_DP) if k != DP_index]
        num_effected_DP = random.choices([0, 1, 2, 3], weights=(5, 35, 50, 10), k=1)[0]
        effected_DP = random.sample(possible_effected_DP, k=num_effected_DP)
        signed_DP = random.choices([1, -1], weights=(60, 40), k=num_effected_DP)

        for j in range(num_effected_DP):
            path_DP[i][0][effected_DP[j]] = signed_DP[j]

    return path_DP
# seed_number = 17 엄청 오래 걸림
# seed_number = 20 conflict 최소가 3
def calculate_conflict_combination(Combination, listing=False):
    Full_list = []
    for combination in Combination:
        full_list = []
        for j in range(num_DP):
            total_index = 0
            for k in [0, 1, 2, 3]:
                total_index += combination[k][j]
            full_list.append(int(np.sign(total_index)))
        Full_list.append(full_list)

    conflict_degree = 0
    for i in range(len(Full_list[0])):
        negative_sign = 0
        positive_sign = 0

        for j in range(len(Full_list)):
            if np.sign(Full_list[j][i]) == -1:
                negative_sign += 1
            elif np.sign(Full_list[j][i]) == 1:
                positive_sign += 1
        #         print("positive_sign : " , positive_sign)
        #         print("negative_sign : ", negative_sign)

        #         print("_________________")
        if (negative_sign > 0) and (positive_sign > 0):
            conflict_degree += 1
    #     print("conflcit_degree :", conflict_degree)

    return conflict_degree, Combination

for seed_number in range(30, 50):
    random.seed(seed_number)
    break_output = 0

    random.seed(seed_number)

    PFO = PFO_make(num_FR, num_option)
    OD = OD_make(num_option, num_DP)

    FR_index = ["FR_{}".format(i + 1) for i in range(num_FR)]
    Opt_index = ["Opt_{}".format(i + 1) for i in range(num_option)]
    DP_index = ["DP_{}".format(i + 1) for i in range(num_DP)]

    random.seed(seed_number)

    # 각 path중 하나를 골라서 path_matrix 만들기

    random.seed(seed_number)
    path_DP_list = []
    for i in range(num_DP):
        num_path = random.randint(1, 3)
        path_DP_list.append(path_make(i))

    import itertools

    # 가능한 모든 조합 생성
    combinations = list(itertools.product(*path_DP_list))

    # PM matrix 만들기
    PM_list = []
    for i in combinations:
        PM_list.append(make_PM(i))

    # phase 1-1 FR - Option
    Z = []
    for i in range(len(PFO)):
        Z.append([int((k + 1) * PFO[i][k]) for k in range(len(PFO[i])) if PFO[i][k] != 0])
    combinations = list(itertools.product(*Z))

    Possible_Option_Candidates = []
    for k in combinations:
        kk = list(k)
        kkk = [abs(j) for j in kk]
        if len(kkk) == len(set(kkk)):
            Possible_Option_Candidates.append(kk)

    Initated_DP_Candidates = []
    Initated_OP_Candidates = []
    for i in Possible_Option_Candidates:
        A, B = checking_DP(i)
        if A != False:
            Initated_DP_Candidates.append(A)
            Initated_OP_Candidates.append(B)

            # 경로 중 conflict 가장 적은 해 찾기
    import time
    import sys

    num_min = 0
    min_conflict = 10000
    conflict_list = []
    combination_list = []
    total_conflict_set = []

    start = time.time()

    for i in range(len(Initated_DP_Candidates)):
        Full_combination = []
        print("{}/{} START".format(i + 1, len(Initated_DP_Candidates)))
        if (min_conflict == 0) and (num_min >= 10):
            end = time.time()
            print("There is too much way with conflict 0")
            break

        DP_list = one_hot_encoding(Initated_DP_Candidates[i])
        imme_list = []
        path_list = []

        for j in range(len(DP_list)):
            print("    {}/{} START".format(j + 1, len(DP_list)))
            first_prop = []
            Second_prop = []
            Third_prop = []
            imme_list = []

            # First prop
            for path_matrix in PM_list:
                first_prop.append(np.dot(DP_list[j], path_matrix))

            first_prop = np.unique(first_prop, axis=0)
            first_prop = first_prop.reshape(len(first_prop), -1)
            for first in first_prop:
                second_prop = []

                for path_matrix in PM_list:
                    second_prop.append(np.dot(first, path_matrix))

                second_prop = np.unique(second_prop, axis=0)

                second_prop = second_prop.reshape(len(second_prop), -1)

                Second_prop.append(second_prop)

            second_prop = []
            # Second prop
            Third_prop = []

            for Second in Second_prop:
                for second in Second:
                    third_prop = []
                    for path_matrix in PM_list:
                        third_prop.append(np.dot(second, path_matrix))

                    third_prop = np.unique(third_prop, axis=0)

                    third_prop = third_prop.reshape(len(third_prop), -1)

                    Third_prop.append(third_prop)

            third_prop = []
            ## propagation

            indexing_third = 0
            for first_index in range(len(first_prop)):
                for second_index in range(len(Second_prop[first_index])):
                    for third_index in range(len(Third_prop[indexing_third])):
                        imme_list.append([DP_list[j],
                                          first_prop[first_index],
                                          Second_prop[first_index][second_index],
                                          Third_prop[indexing_third][third_index]])
                    indexing_third += 1
            path_list.append(imme_list)
        print("{}/{} Full_combination START".format(i + 1, len(Initated_DP_Candidates)))
        Full_combination = list(itertools.product(*path_list))
        print("Full_combination : ".format(len(Full_combination)))
        imme_list = []

        # print("FULL_COMBINATION : ", len(Full_combination))
        start_index = -1
        com_start = time.time()
        for full_combination in Full_combination:
            com_end = time.time()
            if (com_end - com_start) >= 900:
                break

            start_index += 1
            if start_index % 100000 == 0:
                print("{}/{} Full_combination progressing".format(start_index, len(Full_combination)))
            conflict_degree, combination = calculate_conflict_combination(full_combination)
            if conflict_degree not in total_conflict_set:
                total_conflict_set.append(conflict_degree)
            if min_conflict > conflict_degree:
                conflict_list = [conflict_degree]
                combination_list = [combination]
                min_conflict = conflict_degree
                num_min = 1


            elif (min_conflict == conflict_degree) and num_min <= 10:
                conflict_list.append(conflict_degree)
                combination_list.append(combination)
                num_min += 1

            if (min_conflict == 0) and (num_min >= 10):
                end = time.time()
                print(round(end - start, 2))
                break_output = 1
                Full_combination = []
                break
    end = time.time()

    print("__________RESULT_______________________")
    print("seed number : ", seed_number)
    print("TOTAL Full_combination : ", len(Full_combination))
    print("TIME : ", round(end - start, 2))
    print(conflict_list)
    print(total_conflict_set)
    time_calculation.append(round(end - start, 2))
    conflict_seed.append(conflict_list)



