from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
import statistics
import random
import numpy as np
from scipy.optimize import linprog
from helpers import *


p_min = 5
p_max = 1000
w_upper_bound = 0.005 # Each item’s weight is very small, this is the upper bound
w_lower_bound = 0.002
item_amount = 1000


weight_dim=2
runs = 200


def calculate_beta():
    return 1 / (1 + np.log(p_max/p_min))

def calculate_optimal_ratio(y_t):
    beta = calculate_beta()
    #print("beta,",beta)
    #print("y_t",y_t)
    if 0<= y_t and y_t < beta:
        return p_min
    elif y_t <= 1:
        return p_min * np.exp(y_t/beta - 1)
    else:
        return -1

def calculate_alg_return(item_value_list,item_weight_list):
     # Algorithm starts.
    y = 0
    take_list = [0 for i in range(len(item_value_list))]
    for t in range(item_amount):
        #print(t)
        threshold = calculate_optimal_ratio(y)
        if threshold<0:
            break
        #print("threshold",threshold)
        ratio = item_value_list[t] / item_weight_list[t]

        if ratio > threshold:
            take_list[t] = 1

        #print(len(item_value_list),len(take_list))
        y = np.dot(item_weight_list, take_list)
        if y > 1:
            take_list[t] = 0
            break

    alg_result = np.dot(item_value_list, take_list)  
    return alg_result


def generate_random_weight_array(weight_dim):
    # Assume the average weight is bounded by 0.002,0.005
    
    random_array = []
    for each in range(weight_dim):
        random_array.append(random.uniform(w_lower_bound,w_upper_bound))

    return random_array

def experiment(runs):
    ratio_list = []
    for each_run in range(runs):

        # Initialize items.
        item_value_list = []
        item_weight_list = []

        for each_item in range(item_amount):
            weight = generate_random_weight_array(weight_dim)
            
            item_weight_list.append(np.sum(weight))
            value_lower_bound = p_min*np.average(weight)
            value_upper_bound = p_max*np.average(weight)
            value = random.uniform(value_lower_bound,value_upper_bound)
            item_value_list.append(value)

        alg_result = calculate_alg_return(item_value_list,item_weight_list)
        
        optimal_result = solve_lp(item_value_list, item_weight_list)

        ratio = optimal_result / alg_result  
        ratio_list.append(ratio)

    #print(ratio_list)
    plot_runs([i for i in range(runs)], ratio_list,"q3")

#print(numbers_with_sum(8, 5.0) )
# The first question.
experiment(runs)




    

