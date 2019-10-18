import heat as ht
#from heat.utils import matrixgallery
import torch
import numpy as np
import math
from mpi4py import MPI
import os

def projection(u,v):
    #returns projection of v onto vector u
    return (ht.matmul(v, u)/ht.matmul(u,u))*u

def ht_norm(v):
    return math.sqrt(ht.matmul(v , v)._DNDarray__array.item())

class EuklidianDistance():
    def __init__(self):
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def __call__(self, X,Y):
        # X and Y are torch tensors
        k1, f1 = X.shape
        k2, f2 = Y.shape
        if(f1 != f2):
            raise RuntimeError("X and Y have differing feature dimensions (dim = 1), should be equal, but are {} and {}".format(f1,f2))

        Xd = X.unsqueeze(dim=1)
        Yd = Y.unsqueeze(dim=0)
        result = torch.zeros((k1, k2), dtype=torch.float64)

        for i in range(Xd.shape[0]):
            result[i,:]=((Yd - Xd[i, :, :])**2).sum(dim=-1).sqrt()

        #Y_prime = torch.zeros((k1,f1), dtype=Y.dtype)
        #for i in range(0,k2):
        #    Y_prime[:,:]=Y[i,:]
        #    result[:,i] = self.pdist(X, Y_prime)

        return result

class GaussianDistance():
    def __init__(self, sigma=1.):
        self.sigma = sigma

    def __call__(self, X, Y):
        # X and Y are torch tensors
        k1, f1 = X.shape
        k2, f2 = Y.shape
        if (f1 != f2):
            raise RuntimeError(
                "X and Y have differing feature dimensions (dim = 1), should be equal, but are {} and {}".format(f1,
                                                                                                                 f2))

        result = torch.zeros((k1, k2), dtype=torch.float64)
        Y_prime = torch.zeros((k1, f1), dtype=Y.dtype)
        for i in range(0, k2):
            Y_prime[:, :] = Y[i, :]
            result[:, i] = self.__gaussian_distance(X, Y_prime)

        return result

    def Get_Sigma(self):
        return self.sigma

    def __gaussian_distance(self, x1, x2):
        gaussian = lambda x: torch.exp(-x/(2*self.sigma*self.sigma))

        diff = x1 - x2
        diff_2 = diff * diff
        diff_2 = torch.sum(diff_2, dim=1)
        result = gaussian(diff_2)

        return result



def similarity(X, Metric=EuklidianDistance()):
    if (X.split is not None)  and (X.split != 0):
        raise NotImplementedError("Feature Splitting is not supported")
    if len(X.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are supported")

    #comm = ht.MPICommunication(MPI.COMM_WORLD)
    comm = X.comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    stat = MPI.Status()

    K, f = X.shape
    k1,_ = X.lshape

    S = ht.zeros((K, K), dtype=ht.float64, split=1)

    counts, displ, _ = comm.counts_displs_shape(X.shape, X.split)
    num_iter = (size+1)//2

    stationary = X._DNDarray__array
    columns = (displ[rank], displ[rank+1] if (rank+1) != size else K)

    # 0th iteration, calculate diagonal
    moving = stationary
    d_ij = Metric(stationary, stationary)
    S[columns[0]:columns[1], columns[0]:columns[1]] = d_ij

    for iter in range(1, num_iter):
        # Send rank's part of the matrix to the next process in a circular fashion
        receiver = (rank + iter) % size
        sender = (rank - iter) % size

        comm.Send(stationary, dest=receiver, tag=iter)

        comm.handle.Probe(source=sender, tag=iter, status=stat)
        count = int(stat.Get_count(MPI.FLOAT)/f)
        moving = torch.zeros((count, f), dtype=torch.float32)
        comm.Recv(moving, source=sender, tag=iter)

        r1 = displ[sender]
        if sender != size-1:
            r2 = displ[sender+1]
        else:
            r2 = K
        rows = (r1,r2)
        d_ij = Metric(stationary, moving)
        S[rows[0]:rows[1], columns[0]:columns[1]] = d_ij.transpose(0,1)


        #sending result back to sender of moving matrix (for symmetry)
        comm.Send(d_ij, dest=sender, tag=iter)

        sr1 = displ[receiver]
        if receiver != size - 1:
            sr2 = displ[receiver + 1]
        else:
            sr2 = K
        srows = (sr1, sr2)
        symmetric = torch.zeros((srows[1]-srows[0], columns[1]-columns[0]), dtype=torch.float64)

        # Receive calculated tile
        comm.Recv(symmetric, source=receiver, tag=iter)
        S[srows[0]:srows[1], columns[0]:columns[1]] = symmetric


    if((size+1)%2 != 0): #  we need one mor iteration for the first n/2 processes

        receiver = (rank + num_iter) % size
        sender = (rank - num_iter) % size
        if receiver < (size//2) :
            comm.Send(stationary, dest=receiver, tag=num_iter)

            #Receiving back result
            sr1 = displ[receiver]
            if receiver != size - 1:
                sr2 = displ[receiver + 1]
            else:
                sr2 = K
            srows = (sr1, sr2)
            symmetric = torch.zeros((srows[1] - srows[0], columns[1] - columns[0]), dtype=torch.float64)

            comm.Recv(symmetric, source=receiver, tag=num_iter)
            S[srows[0]:srows[1], columns[0]:columns[1]] = symmetric

        else:
            pass

        if rank < (size//2):
            comm.handle.Probe(source=sender, tag=num_iter, status=stat)
            count = int(stat.Get_count(MPI.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=num_iter)

            r1 = displ[sender]
            if sender != size - 1:
                r2 = displ[sender + 1]
            else:
                r2 = K
            rows = (r1, r2)

            d_ij = Metric(stationary, moving)
            S[rows[0]:rows[1], columns[0]:columns[1]] = d_ij.transpose(0,1)

            # sending result back to sender of moving matrix (for symmetry)
            comm.Send(d_ij, dest=sender, tag=num_iter)

        else:
            pass

    #S.resplit_(axis=None)
    return S


def laplacian(data, metric, threshold):
    #calculated similarity matrix
    S = similarity(data, metric)
    #print("Similarity Matrix = {}".format(S))
    # derive adjacency matrix from similarity matrix via threshold
    A = ht.int(S > threshold)- ht.eye(S.shape, dtype=ht.int, split=S.split)
    #print("Adjacency Matrix = {}".format(A))

    # calculate degree matrix by summing rows of adjacency matrix
    degree = ht.sum(A, axis=1)
    D = ht.empty(S.shape, dtype=ht.int, split = S.split, device=S.device, comm=S.comm)
    D._DNDarray__array = torch.diag(degree._DNDarray__array)
    #print("Degree Matrix = {}\n".format(D))

    L = D - A

    return L


def lanczos_ht(A, v0, m):
    n, column = A.shape
    if n != column: raise TypeError("Input Matrix A needs to be symmetric.")
    T = ht.zeros((m,m))
    if(A.split == 0) : v = ht.zeros((n,m), split=A.split)
    else: v = ht.zeros((n,m))

    # # 0th iteration
    # # vector v0 has euklidian norm =1
    w = ht.matmul(A, v0)
    alpha = ht.matmul(w, v0)._DNDarray__array.item()
    beta = 0
    w = w - alpha*v0
    T[0,0] = alpha
    v[:,0] = v0

    for i in range(1, m):
        beta = ht_norm(w)
        if beta == 0.:
            #pick a random vector
            print("Lanczos Breakdown in iteration {}\n".format(i))
            vr = ht.random.rand(n)
            # orthogonalize v_r with respect to all vectors v[i]
            for j in range(i):
                vr = vr - projection(v[:,j], vr)
            # normalize v_r to euklidian norm 1 and set as ith vector v
            vi = vr/ht_norm(vr)
        else:
            vi = w/beta

        w = ht.matmul(A, vi)
        alpha = ht.matmul(w, vi)._DNDarray__array.item()
        w = w - alpha*vi - beta*v[:,i-1]
        T[i-1, i] = beta
        T[i, i-1] = beta
        T[i, i] = alpha
        v[:, i] = vi

    if(v.split != None): v.resplit_(axis=None)

    return v, T



def conjgrad(A, b, x):
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r, r)

    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(r, r)
        if np.linalg.norm(rsnew) < 1e-10:
            return x

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x



def conjgrad_heat(A, b, x):
    r = b - ht.matmul(A , x)
    p = r
    rsold = ht.matmul(r , r).item()

    for i in range(len(b)):
        Ap = ht.matmul(A , p)
        alpha = rsold / ht.matmul(p , Ap).item()
        x = x + (alpha * p)
        r = r - (alpha * Ap)
        rsnew = ht.matmul(r , r).item()
        if(math.sqrt(rsnew)< 1e-10):
            print("Residual r = {} reaches tolerance in it = {}".format(math.sqrt(rsnew), i))
            return  x
        p = r + ((rsnew / rsold) * p)
        rsold = rsnew

    return x



if __name__ == "__main__":
    comm2 = ht.MPICommunication(MPI.COMM_WORLD)
    rank = comm2.Get_rank()

    data = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/src/heat/heat/datasets/data/iris.h5'), 'data', split=0)
    k1, f1 = data.shape

    Xd = data.expand_dims(axis=1)
    Yd = data.expand_dims(axis=0)
    result = ht.zeros((k1, k1), dtype=ht.float32)

    for i in range(Xd.shape[0]):
        result[i, :] = ((Yd - Xd[i, :, :]) ** 2).sum(dim=-1).sqrt()


