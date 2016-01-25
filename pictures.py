import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def sample_arr(arr):
    pass

def gradient_disp(arr):
    plt.close()
    plt.imshow(arr, colormap="Greys")
    plt.colorbar()
    plt.savefig("pics/001_grad")

def gradient_abs_disp(arr):
    plt.close()
    plt.imshow(np.abs(arr), colormap="Greys")
    plt.colorbar()
    plt.savefig("pics/002_abs_grad")

def normalized_gradient_disp(arr):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    plt.imshow(new_arr, colormap="Greys")
    plt.colorbar()
    plt.savefig("pics/003_normalized_grad")

def sampled_gradient_disp(arr):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    plt.imshow(new_arr, colormap="Greys")
    plt.colorbar()
    plt.savefig("pics/004_sampled_grad")

def sampled_gradient_degree_disp(arr):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    degrees = new_arr.sum(axis=0)
    plt.loglog(degrees)
    plt.savefig("pics/005_gradient_degrees")

def sample_gradient_otherstats(arr):
    print "other stats: "
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    net = nx.from_numpy_matrix(new_arr)
    print "diameter: ", nx.diameter(net)
    print "mean clustering coeffs", nx.clustering_coefficient(net)

if __name__ == "__main__":
    np_arr = np.load("grad_mat.npy")
    print np_arr
