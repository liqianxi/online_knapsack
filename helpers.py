from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
import statistics
import random
import numpy as np
import cvxpy
def plot_runs(runs_list, ratio_list,name):
    fig, ax = plt.subplots(1, 1)
    ax.plot(runs_list, ratio_list, 'r-', lw=1, alpha=0.6, label='mean')

    plt.title(name)
    plt.xlabel("# of runs")
    ax.legend()
    fig.text(.5, 0.02, "min value:"+str(np.min(ratio_list))+", max value:"+str(np.max(ratio_list)), ha='center')
    fig.set_size_inches(7, 7, forward=True)
    plt.ylabel("OPT / ALG")
    plt.savefig(name+'.png')


def solve_lp(item_value_list, item_weight_list):
    # The data for the Knapsack problem
    # P is total weight capacity of sack
    # weights and utilities are also specified

    # The variable we are solving for
    selection = cvxpy.Variable(len(item_weight_list), boolean = True)
    #cvxpy.Bool(len(weights))

    # The sum of the weights should be less than or equal to P
    weight_constraint = item_weight_list * selection <= 1

    # Our total utility is the sum of the item utilities
    total_utility = item_value_list * selection

    # We tell cvxpy that we want to maximize total utility 
    # subject to weight_constraint. All constraints in 
    # cvxpy must be passed as a list
    knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_utility), [weight_constraint])

    # Solving the problem
    knapsack_problem.solve(solver=cvxpy.GLPK_MI)
    return np.dot(item_value_list, selection.value)