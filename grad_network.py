import numpy as np
import numpy.random as npr
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    FILENAME = "grad_mat.npy"
    network_mat = np.abs(np.load(FILENAME))
    max_network_member = np.max(network_mat)
    print "this is not really something you are supposed to do, I think"
    print "but it is quite interesting."
    print "take the gradient matrix, normalize it so that the biggest single member of that matrix is 1"
    network_mat /= max_network_member
    network = nx.Graph()
    print "that defines an ensemble of networks. sample from that ensemble."
    print "that sample will look suspiciously like a social network."
    print "did you expect that?"
    for x in xrange(network_mat.shape[0]):
        for y in xrange(network_mat.shape[1]):
            # print network_mat[x,y]
            if network_mat[x,y] > npr.rand():
                network.add_edge(x, y, weight=network_mat[x,y])
    degree_sequence = sorted(nx.degree(network).values(),reverse=True)
    plt.hist(degree_sequence, 60)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.ylabel("number of nodes with that degree")
    plt.xlabel("degree")
    plt.axis([1, 2000, 0, 2000])
    plt.grid(True)
    plt.title("degree histogram")
    plt.show()
    print "tail is definitely not power law, but definitely a heavy tail tho"
    print "now let's see that adjacency matrix"
    adj_mat = nx.adjacency_matrix(network)
    plt.imshow(adj_mat)
    plt.colorbar()
    plt.show()
    print "calculating diameter. this is the naive diameter, not the 95% one, so it will take an obnoxious amount of time"
    print "diameter is: ", nx.diameter(network)
    print "mean clustering coefficient: ", np.mean(np.array([val for key, val in nx.clustering(network).iteritems()]))
