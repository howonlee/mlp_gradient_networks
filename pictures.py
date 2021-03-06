import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import networkx as nx
import time

def sample_arr(arr):
    new_arr = np.zeros_like(arr)
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if npr.rand() < arr[x, y]:
                new_arr[x, y] = 1.0
    return new_arr

def gradient_disp(arr, title, filename):
    plt.close()
    plt.imshow(arr, cmap="Greys")
    plt.colorbar()
    plt.title(title)
    plt.savefig("pics/" + filename)

def gradient_abs_disp(arr, title, filename):
    plt.close()
    plt.imshow(np.abs(arr), cmap="Greys")
    plt.colorbar()
    plt.title(title)
    plt.savefig("pics/" + filename)

def normalized_gradient_disp(arr, title, filename):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    plt.imshow(new_arr, cmap="Greys")
    plt.colorbar()
    plt.title(title)
    plt.savefig("pics/" + filename)

def sampled_gradient_disp(arr, title, filename):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    plt.imshow(new_arr, cmap="Greys")
    plt.colorbar()
    plt.title(title)
    plt.savefig("pics/" + filename)

def sampled_gradient_degree_disp(arr, title, filename):
    plt.close()
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    degs = new_arr.sum(axis=0)
    plt.hist(degs)
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("number of nodes with degree")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.savefig("pics/" + filename)

def sampled_gradient_otherstats(arr):
    print "other stats: "
    new_arr = np.abs(arr)
    new_arr /= np.max(new_arr)
    new_arr = sample_arr(new_arr)
    arr_net = nx.from_numpy_matrix(new_arr[:-1, :])
    giant_component_net = arr_net.subgraph(max(nx.connected_components(arr_net), key=len))
    print time.clock()
    diameter = nx.diameter(giant_component_net)
    print time.clock()
    clustering_coeff = nx.average_clustering(arr_net)
    print "diameter: ", diameter
    print "clustering coeff: ", clustering_coeff

if __name__ == "__main__":
    np_arr = np.load("grad_mat.npy")
    # gradient_disp(np_arr, "Gradient", "a001_grad")
    # gradient_abs_disp(np_arr, "Absolute Gradient", "a002_abs_grad")
    # normalized_gradient_disp(np_arr, "Normalized Absolute Gradient", "a003_norm_grad")
    # sampled_gradient_disp(np_arr, "Sampled Gradient Network", "a004_grad_net")
    # sampled_gradient_degree_disp(np_arr, "Gradient Network Degree Distribution", "e005_grad_deg")
    # sampled_gradient_otherstats(np_arr)
