import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import networkx as nx

def sample_arr(arr):
    new_arr = np.zeros_like(arr)
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if npr.rand() < arr[x, y]:
                new_arr[x, y] = 1.0
    return new_arr

def gradient_disp(arr):
    plt.close()
    plt.imshow(arr, cmap="Greys")
    plt.colorbar()
    plt.savefig("pics/001_grad")

def gradient_abs_disp(arr):
    plt.close()
    plt.imshow(np.abs(arr), cmap="Greys")
    plt.colorbar()
    plt.savefig("pics/002_abs_grad")

def normalized_gradient_disp(arr):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    plt.imshow(new_arr, cmap="Greys")
    plt.colorbar()
    plt.savefig("pics/003_normalized_grad")

def sampled_gradient_disp(arr):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    plt.imshow(new_arr, cmap="Greys")
    plt.colorbar()
    plt.savefig("pics/004_sampled_grad")

def sampled_gradient_degree_disp(arr):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    degrees = sorted(new_arr.sum(axis=0), reverse=True)
    plt.loglog(degrees)
    plt.savefig("pics/005_gradient_degrees")

# def sample_gradient_otherstats(arr):
#     print "other stats: "
#     new_arr = np.abs(arr)
#     new_arr /= np.max(new_arr)
#     new_arr = sample_arr(new_arr)
#     net = nx.from_numpy_matrix(new_arr)
#     print "diameter: ", nx.diameter(net)
#     print "mean clustering coeffs", nx.clustering_coefficient(net)

if __name__ == "__main__":
    np_arr = np.load("grad_mat.npy")
    gradient_disp(np_arr)
    gradient_abs_disp(np_arr)
    normalized_gradient_disp(np_arr)
    sampled_gradient_disp(np_arr)
    sampled_gradient_degree_disp(np_arr)
