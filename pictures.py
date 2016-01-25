import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def gradient_disp(arr):
    plt.close()
    plt.savefig("pics/001_grad")

def gradient_abs_disp(arr):
    plt.close()
    plt.savefig("pics/002_abs_grad")

def normalized_gradient_disp(arr):
    plt.close()
    plt.savefig("pics/003_normalized_grad")

def sampled_gradient_disp(arr):
    plt.close()
    plt.savefig("pics/004_sampled_grad")

def sampled_gradient_degree_disp(arr):
    plt.close()
    plt.savefig("pics/005_gradient_degrees")

def sample_gradient_otherstats(arr):
    print "other stats: "

if __name__ == "__main__":
    np_arr = np.load("grad_mat.npy")
    print np_arr
