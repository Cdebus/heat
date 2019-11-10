import heat as ht
#from heat.utils import matrixgallery
import torch
import numpy as np
import math
import os
import time


from mpi4py import MPI

class Timing:
    def __init__(self,name,verbosity=1):
        self.verbosity = verbosity
        self.name = name

    def __enter__(self):
        if self.verbosity > 0:
            self.start = time.perf_counter()

    def __exit__(self,*args):
        if self.verbosity > 0:
            stop = time.perf_counter()
            print("Time {}: {:4.4f}s".format(self.name,stop-self.start))

def projection(u,v):
    #returns projection of v onto vector u
    return (ht.dot(v, u)/ht.dot(u,u))*u

def ht_norm(v):
    return math.sqrt(ht.dot(v , v))

class EuclidianDistance():
    def __init__(self):
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.name = "Pairwise Euclidian Distance"

    def __call__(self, X,Y):
        # X and Y are torch tensors

        #start = time.perf_counter()
        k1, f1 = X.shape
        k2, f2 = Y.shape
        if(f1 != f2):
            raise RuntimeError("X and Y have differing feature dimensions (dim = 1), should be equal, but are {} and {}".format(f1,f2))

        Xd = X.unsqueeze(dim=1)
        Yd = Y.unsqueeze(dim=0)
        result = torch.zeros((k1, k2), dtype=torch.float64)

        for i in range(Xd.shape[0]):
            result[i,:] = ((Yd - Xd[i,:,:]) ** 2).sum(dim=-1).sqrt()

        return result
    def Get_Name(self):
        return self.name

    def print_self(self):
        "Metric {} ".format(self.name)


class GaussianDistance():
    def __init__(self, sigma=1.):
        self.sigma = sigma
        self.name = "Pairwise Gaussian Kernel"

    def __call__(self, X, Y):
        # X and Y are torch tensors
        k1, f1 = X.shape
        k2, f2 = Y.shape
        if (f1 != f2):
            raise RuntimeError(
                "X and Y have differing feature dimensions (dim = 1), should be equal, but are {} and {}".format(f1,
                                                                                                                 f2))
        Xd = X.unsqueeze(dim=1)
        Yd = Y.unsqueeze(dim=0)
        result = torch.zeros((k1, k2), dtype=torch.float64)
        for i in range(Xd.shape[0]):
            result[i, :] = torch.exp(-((Yd - Xd[i,:,:]) ** 2).sum(dim=-1)/(2*self.sigma*self.sigma))

        return result

    def Get_Sigma(self):
        return self.sigma

    def Get_Name(self):
        return self.name

    def print_self(self):
        return "Metric {} with parameter sigma = {}".format(self.name, self.sigma)

    def __gaussian_distance(self, x1, x2):
        gaussian = lambda x: torch.exp(-x/(2*self.sigma*self.sigma))

        diff = x1 - x2
        diff_2 = diff * diff
        diff_2 = torch.sum(diff_2, dim=1)
        result = gaussian(diff_2)

        return result



def similarity_piecewise(X, Metric=EuclidianDistance(), limit = 2000000):
    if (X.split is not None)  and (X.split != 0):
        raise NotImplementedError("Feature Splitting is not supported")
    if len(X.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are supported")

    comm = X.comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    K, f = X.shape
    k1,f1 = X.lshape

    S = ht.zeros((K, K), dtype=ht.float64, split=0)

    counts, displ, _ = comm.counts_displs_shape(X.shape, X.split)
    num_iter = (size+1)//2

    stationary = X._DNDarray__array
    rows = (displ[rank], displ[rank+1] if (rank+1) != size else K)

    # 0th iteration, calculate diagonal
    d_ij = Metric(stationary, stationary)
    S[rows[0]:rows[1], rows[0]:rows[1]] = d_ij

    for iter in range(1, num_iter):
        # Send rank's part of the matrix to the next process in a circular fashion
        receiver = (rank + iter) % size
        sender = (rank - iter) % size

        col1 = displ[sender]
        if sender != size-1:
            col2 = displ[sender+1]
        else:
            col2 = K
        columns = (col1, col2)

        ######################### All but the first iter processes are receiving, then sending#######################################
        if((rank //iter) != 0):
            num_recv = torch.tensor(0)
            comm.Recv(num_recv, source=sender, tag=iter)
            num_recv = num_recv.item()
            count = int(num_recv / f1)
            moving = torch.zeros((count, f1), dtype=torch.float32)

            if num_recv > limit:
                num_chunks = math.ceil(num_recv / limit)

                cnt = np.full((num_chunks,), count // num_chunks)
                cnt[:count % num_chunks] += 1
                dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
                np.cumsum(cnt[:-1], out=dsp[1:])

                for chunk in range(num_chunks):
                    slice1 = dsp[chunk]
                    if chunk != (num_chunks - 1):
                        slice2 = dsp[chunk + 1]
                    else:
                        slice2 = count

                    temp_buf = torch.zeros((slice2 - slice1, f), dtype=torch.float32)
                    comm.Recv(temp_buf, source=sender, tag=iter*100+chunk)
                    # print("Receiving  {} ".format(temp_buf))
                    moving[slice1:slice2, :] = temp_buf
            else:
                comm.Recv(moving, source=sender, tag=iter*100)


        ######################### Sending to next Process #######################################
        num_send = k1*f1
        comm.Send(torch.tensor(num_send), dest=receiver, tag=iter)

        if num_send > limit:
            num_chunks = math.ceil(num_send/limit)
            cnt = np.full((num_chunks,), k1 // num_chunks)
            cnt[:k1 % num_chunks] += 1
            dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
            np.cumsum(cnt[:-1], out=dsp[1:])

            for chunk in range(num_chunks):
                slice1 = dsp[chunk]
                if chunk != (num_chunks-1):
                    slice2 = dsp[chunk+1]
                else:
                    slice2 = k1
                sendbuf = stationary[slice1:slice2, :].clone()
                comm.Send(sendbuf, dest=receiver, tag=iter*100+chunk)
        else:
            comm.Ssend(stationary, dest=receiver, tag=iter*100)

        ######################### The first iter processes can now receive after seding#######################################

        if((rank //iter) == 0):
            num_recv = torch.tensor(0)
            comm.Recv(num_recv, source=sender, tag=iter)
            num_recv = num_recv.item()
            count = int(num_recv / f1)
            moving = torch.zeros((count, f1), dtype=torch.float32)

            if num_recv > limit:
                num_chunks = math.ceil(num_recv / limit)

                cnt = np.full((num_chunks,), count // num_chunks)
                cnt[:count % num_chunks] += 1
                dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
                np.cumsum(cnt[:-1], out=dsp[1:])

                for chunk in range(num_chunks):
                    slice1 = dsp[chunk]
                    if chunk != (num_chunks - 1):
                        slice2 = dsp[chunk + 1]
                    else:
                        slice2 = count

                    temp_buf = torch.zeros((slice2 - slice1, f), dtype=torch.float32)
                    comm.Recv(temp_buf, source=sender, tag=iter*100+chunk)
                    moving[slice1:slice2, :] = temp_buf
            else:
                comm.Recv(moving, source=sender, tag=iter*100)

        d_ij = Metric(stationary, moving)
        S[rows[0]:rows[1], columns[0]:columns[1]] = d_ij

        scol1 = displ[receiver]
        if receiver != size - 1:
            scol2 = displ[receiver + 1]
        else:
            scol2 = K

        scolumns = (scol1, scol2)
        symmetric = torch.zeros((rows[1]-rows[0], scolumns[1]-scolumns[0]), dtype=torch.float64)

        if((rank //iter) != 0):
            comm.Recv(symmetric, source=receiver, tag=size + iter)

        comm.Send(d_ij, dest=sender, tag=size+iter)

        if((rank //iter) == 0):
            comm.Recv(symmetric, source=receiver, tag=size + iter)

        # Receive calculated tile
        S[rows[0]:rows[1], scolumns[0]:scolumns[1]] = symmetric

    if((size+1)%2 != 0): #  we need one mor iteration for the first n/2 processes

        receiver = (rank + num_iter) % size
        sender = (rank - num_iter) % size

        # Case 1: only receiving
        if rank < (size//2):
            ############################### Receiving data from sender process #################
            num_recv = torch.tensor(0)
            comm.Recv(num_recv, source=sender, tag=num_iter)
            num_recv = num_recv.item()
            count = int(num_recv / f1)
            moving = torch.zeros((count, f1), dtype=torch.float32)

            if num_recv > limit:
                num_chunks = math.ceil(num_recv / limit)

                cnt = np.full((num_chunks,), count // num_chunks)
                cnt[:count % num_chunks] += 1
                dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
                np.cumsum(cnt[:-1], out=dsp[1:])

                for chunk in range(num_chunks):
                    slice1 = dsp[chunk]
                    if chunk != (num_chunks - 1):
                        slice2 = dsp[chunk + 1]
                    else:
                        slice2 = count

                    temp_buf = torch.zeros((slice2 - slice1, f), dtype=torch.float32)
                    comm.Recv(temp_buf, source=sender, tag=num_iter * 100 + chunk)
                    # print("Receiving  {} ".format(temp_buf))
                    moving[slice1:slice2, :] = temp_buf
            else:
                comm.Recv(moving, source=sender, tag=num_iter * 100)

            ############################### Calculating Tile #################
            col1 = displ[sender]
            if sender != size - 1:
                col2 = displ[sender + 1]
            else:
                col2 = K
            columns = (col1, col2)

            d_ij = Metric(stationary, moving)
            S[rows[0]:rows[1], columns[0]:columns[1]] = d_ij

            ################################ sending result back to sender ###############################
            comm.Send(d_ij, dest=sender, tag=num_iter)


        # Case 2 : only sending processes
        else :
            ######################### Sending to next Process #######################################
            num_send = k1 * f1
            comm.Send(torch.tensor(num_send), dest=receiver, tag=num_iter)

            if num_send > limit:
                num_chunks = math.ceil(num_send / limit)
                cnt = np.full((num_chunks,), k1 // num_chunks)
                cnt[:k1 % num_chunks] += 1
                dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
                np.cumsum(cnt[:-1], out=dsp[1:])

                for chunk in range(num_chunks):
                    slice1 = dsp[chunk]
                    if chunk != (num_chunks - 1):
                        slice2 = dsp[chunk + 1]
                    else:
                        slice2 = k1
                    sendbuf = stationary[slice1:slice2, :].clone()
                    comm.Send(sendbuf, dest=receiver, tag=num_iter * 100 + chunk)
            else:
                comm.Ssend(stationary, dest=receiver, tag=num_iter * 100)


            ################################ Receiving result back  ###############################
            scol1 = displ[receiver]
            if receiver != size - 1:
                scol2 = displ[receiver + 1]
            else:
                scol2 = K
            scolumns = (scol1, scol2)
            symmetric = torch.zeros((rows[1] - rows[0], scolumns[1] - scolumns[0]), dtype=torch.float64)
            comm.Recv(symmetric, source=receiver, tag=num_iter)
            S[rows[0]:rows[1], scolumns[0]:scolumns[1]] = symmetric

    return S


def similarity(X, Metric=EuclidianDistance()):
    if (X.split is not None)  and (X.split != 0):
        raise NotImplementedError("Feature Splitting is not supported")
    if len(X.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are supported")

    comm = X.comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    K, f = X.shape
    k1, _ = X.lshape

    S = ht.zeros((K, K), dtype=ht.float64, split=0)

    counts, displ, _ = comm.counts_displs_shape(X.shape, X.split)
    num_iter = (size + 1) // 2

    stationary = X._DNDarray__array
    rows = (displ[rank], displ[rank + 1] if (rank + 1) != size else K)

    # 0th iteration, calculate diagonal
    d_ij = Metric(stationary, stationary)
    S[rows[0]:rows[1], rows[0]:rows[1]] = d_ij

    for iter in range(1, num_iter):
        if(rank == 0): print(" ####### Round {} #######".format(iter))

        # Send rank's part of the matrix to the next process in a circular fashion
        receiver = (rank + iter) % size
        sender = (rank - iter) % size

        col1 = displ[sender]
        if sender != size - 1:
            col2 = displ[sender + 1]
        else:
            col2 = K
        columns = (col1, col2)

        ######################### All but the first iter processes are receiving, then sending#######################################
        if ((rank // iter) != 0):
            stat = MPI.Status()
            comm.handle.Probe(source=sender, tag=iter, status=stat)
            count = int(stat.Get_count(MPI.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=iter)

        ######################### Sending to next Process #######################################
        comm.Send(stationary, dest=receiver, tag=iter)

        ######################### The first iter processes can now receive after seding#######################################
        if ((rank // iter) == 0):
            stat = MPI.Status()
            comm.handle.Probe(source=sender, tag=iter, status=stat)
            count = int(stat.Get_count(MPI.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=iter)

        d_ij = Metric(stationary, moving)
        S[rows[0]:rows[1], columns[0]:columns[1]] = d_ij


        # Receive calculated tile
        scol1 = displ[receiver]
        if receiver != size - 1:
            scol2 = displ[receiver + 1]
        else:
            scol2 = K
        scolumns = (scol1, scol2)
        symmetric = torch.zeros(scolumns[1] - scolumns[0], (rows[1] - rows[0]), dtype=torch.float64)
        # Receive calculated tile
        if((rank //iter) != 0):
            comm.Recv(symmetric, source=receiver, tag=iter)

        # sending result back to sender of moving matrix (for symmetry)
        comm.Send(d_ij, dest=sender, tag=iter)
        if((rank //iter) == 0):
            comm.Recv(symmetric, source=receiver, tag=iter)
        S[rows[0]:rows[1], scolumns[0]:scolumns[1]] = symmetric.transpose(0,1)

    if ((size + 1) % 2 != 0):  # we need one mor iteration for the first n/2 processes
        receiver = (rank + num_iter) % size
        sender = (rank - num_iter) % size

        # Case 1: only receiving
        if rank < (size // 2):
            stat = MPI.Status()
            comm.handle.Probe(source=sender, tag=num_iter, status=stat)
            count = int(stat.Get_count(MPI.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=num_iter)

            col1 = displ[sender]
            if sender != size - 1:
                col2 = displ[sender + 1]
            else:
                col2 = K
            columns = (col1, col2)

            d_ij = Metric(stationary, moving)
            S[rows[0]:rows[1], columns[0]:columns[1]] = d_ij
            # sending result back to sender of moving matrix (for symmetry)
            comm.Send(d_ij, dest=sender, tag=num_iter)

        # Case 2 : only sending processes
        else:
            comm.Send(stationary, dest=receiver, tag=num_iter)

            # Receiving back result
            scol1 = displ[receiver]
            if receiver != size - 1:
                scol2 = displ[receiver + 1]
            else:
                scol2 = K
            scolumns = (scol1, scol2)
            symmetric = torch.zeros((scolumns[1] - scolumns[0], rows[1] - rows[0]), dtype=torch.float64)
            comm.Recv(symmetric, source=receiver, tag=num_iter)
            S[rows[0]:rows[1], scolumns[0]:scolumns[1]] = symmetric.transpose(0,1)

    return S

def unnormalized_laplacian_fullyConnected(S):

    degree = ht.sum(S, axis=1)
    D = ht.zeros(S.shape, dtype=ht.int, split = S.split)

    counts, displ, _ = D.comm.counts_displs_shape(D.shape, D.split)
    c1 = displ[rank]
    if rank != D.comm.size-1:
        c2 =displ[rank+1]
    else:
        c2 = D.shape[1]

    D._DNDarray__array[:,c1:c2] = torch.diag(degree._DNDarray__array)
    L = D - S
    return L

def normalized_laplacian_fullyConnected(S):

    d = ht.sqrt(1./ht.sum(S, axis=1))
    D_ = ht.zeros(S.shape, split = S.split)

    counts, displ, _ = D_.comm.counts_displs_shape(D_.shape, D_.split)
    c1 = displ[rank]
    if rank != D_.comm.size-1:
        c2 =displ[rank+1]
    else:
        c2 = D_.shape[1]

    D_._DNDarray__array[:,c1:c2] = torch.diag(d._DNDarray__array)
    L_sym = ht.eye(S.shape, split=S.split) - ht.matmul(D_, ht.matmul(S, D_))
    return L_sym

def unnormalized_laplacian_eNeighbour(S, epsilon):

    A = ht.int(S < epsilon) - ht.eye(S.shape, dtype=ht.int, split=S.split)
    degree = ht.sum(A, axis=1)
    D = ht.zeros(A.shape, dtype=ht.int, split = A.split)

    counts, displ, _ = D.comm.counts_displs_shape(D.shape, D.split)
    c1 = displ[rank]
    if rank != D.comm.size-1:
        c2 =displ[rank+1]
    else:
        c2 = D.shape[1]
    D._DNDarray__array[:,c1:c2] = torch.diag(degree._DNDarray__array)
    L = D - A
    return L

def normalized_laplacian_eNeighbour(S,epsilon):

    A = ht.int(S < epsilon) - ht.eye(S.shape, dtype=ht.int, split=S.split)

    d = ht.sqrt(1./ht.sum(A, axis=1))
    D_ = ht.zeros(A.shape, split = A.split)

    counts, displ, _ = D_.comm.counts_displs_shape(D_.shape, D_.split)
    c1 = displ[rank]
    if rank != D_.comm.size-1:
        c2 =displ[rank+1]
    else:
        c2 = D_.shape[1]

    D_._DNDarray__array[:,c1:c2] = torch.diag(d._DNDarray__array)
    L_sym = ht.eye(A.shape, split=A.split) - ht.matmul(D_, ht.matmul(A, D_))
    return L_sym

def lanczos_ht(A, m, v0= None):

    n, column = A.shape
    if n != column:
        raise TypeError("Input Matrix A needs to be symmetric.")
    T = ht.zeros((m, m))
    if(A.split == 0) :
        v = ht.zeros((n,m), split=A.split, dtype=ht.float64)
    else:
        v = ht.zeros((n,m), dtype=ht.float64)

    if v0 is None:
        vr = ht.random.rand(n)
        v0 = vr / ht_norm(vr)

    # # 0th iteration
    # # vector v0 has euklidian norm =1
    w = ht.matmul(A, v0)
    alpha = ht.dot(w, v0)
    beta = 0
    w = w - alpha*v0
    T[0,0] = alpha
    v[:,0] = v0

    for i in range(1, m):
        beta = ht_norm(w)
        if beta == 0.:
            #pick a random vector
            print("Lanczos Breakdown in iteration {}".format(i))
            vr = ht.random.rand(n)
            # orthogonalize v_r with respect to all vectors v[i]
            for j in range(i):
                vr = vr - projection(v[:,j], vr)
            # normalize v_r to euklidian norm 1 and set as ith vector v
            vi = vr/ht_norm(vr)
        else:
            vi = w/beta

        w = ht.matmul(A, vi)
        alpha = ht.dot(w, vi)
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
    rsold = ht.matmul(r , r)._DNDarray__array.item()

    for i in range(len(b)):
        Ap = ht.matmul(A , p)
        alpha = rsold / ht.matmul(p , Ap)._DNDarray__array.item()
        x = x + (alpha * p)
        r = r - (alpha * Ap)
        rsnew = ht.matmul(r , r)._DNDarray__array.item()
        if(math.sqrt(rsnew)< 1e-10):
            print("Residual r = {} reaches tolerance in it = {}".format(math.sqrt(rsnew), i))
            return  x
        p = r + ((rsnew / rsold) * p)
        rsold = rsnew

    return x

def test_Sending():
    stat = MPI.Status()
    print("Rank {} up and Running".format(rank))

    #data_ht = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=0)

    n = 300
    limit = 2000000
    data_ht = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/src/heat-Phillip/heat/datasets/data/snapshot_matrix_test289.h5'), 'snapshots')
    sample = data_ht[:n, :]

    k, f = sample.shape

    if rank == 0:
        print("data loaded")


    start = time.perf_counter()
    if rank == 0:
        num_send = sample.shape[0] * sample.shape[1]

        comm.Send(torch.tensor(num_send), dest=1, tag=999)
        print("Attempting to send {} slices, with total number of entries = {}".format(n, num_send))

        if num_send > limit:
            num_chunks = math.ceil(num_send/limit)
            cnt = np.full((num_chunks,), n // num_chunks)
            cnt[:n % num_chunks] += 1
            dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
            np.cumsum(cnt[:-1], out=dsp[1:])
            print("Send data too big, will be send in {} chunks consisting {} entries at positions {} ".format(num_chunks, cnt, dsp))



            for chunk in range(num_chunks):
                s1 = dsp[chunk]
                if chunk != (num_chunks-1):
                    s2 = dsp[chunk+1]
                else:
                    s2 = n
                sendbuf = sample[s1:s2, :].copy()
                print("Sending chunk {}, slices {}:{}".format(chunk, s1, s2))
                comm.Send(sendbuf, dest=1, tag=chunk)

        else:
            comm.Ssend(sample, dest=1, tag=n)

    else:
        moving = torch.zeros((n, f), dtype=torch.float32)

        num_recv = torch.tensor(0)
        comm.Recv(num_recv, source=0, tag=999)
        num_recv = num_recv.item()
        print("Attempting to Receive {} slices, with total number of entries = {}".format(n, num_recv))

        if num_recv > limit :
            num_chunks = math.ceil(num_recv/limit)

            cnt = np.full((num_chunks,), n // num_chunks)
            cnt[:n % num_chunks] += 1
            dsp = np.zeros((num_chunks,), dtype=cnt.dtype)
            np.cumsum(cnt[:-1], out=dsp[1:])
            print(" Data too big, will be received in {} chunks consisting {} entries at positions {} ".format(num_chunks, cnt, dsp))


            for chunk in range(num_chunks):
                s1 = dsp[chunk]
                if chunk != (num_chunks-1):
                    s2 = dsp[chunk+1]
                else:
                    s2 = n

                temp_buf = torch.zeros((s2-s1, f), dtype=torch.float32)
                print("Receiving chunk {}, slices {}:{}".format(chunk, s1, s2))
                comm.Recv(temp_buf, source=0, tag=chunk)
                moving[s1:s2, :] = temp_buf


        else:
            comm.Recv(moving, source=0, tag=n)


        stop = time.perf_counter()
        print("Process {} received data of size {}  after  {}s".format(rank, moving.shape, stop - start))


if __name__ == "__main__":
    ############################ Initialization ############################
    comm = ht.MPICommunication(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print("Process {} running on {}, with {} torch threads".format(rank, name, torch.get_num_threads()))

    metric = GaussianDistance(sigma=5E4)
    m = 500 # Lanczos iterations


    ############################ Data Loading ############################
    data_ht = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/data/snapshot_matrix_296.h5'),'snapshots', split=0)
    #data_ht = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/src/heat/heat/datasets/data/iris.h5'), 'data', split=0)
    #S = ht.load_hdf5('/home/debu_ch/src/heat/results/Test248/Similarity_Gaussian.h5', 'GaussianKernel', split=0)
    if rank == 0:
        print("Data loaded: snapshot_matrix_296.h5")
        start = time.perf_counter()

    ############################ Spectral clustering pipeline ############################
    # 1. Calculation of Similarity Matrix
    S = similarity(data_ht, metric) #Gaussian Kernel
    #S_E = similarity(data_ht) #Euclidian Distance
    ht.save_hdf5(S, '/home/debu_ch/result/results/Test296/Similarity_Run1.h5', 'GaussianKernel')
    #ht.save_hdf5(S_E, '/home/debu_ch/src/heat/results/Test248/Similarity_Euclidian.h5', 'EuclidianDistance')

    if rank == 0:
        stop = time.perf_counter()
        print("Calculation of Similarity Matrix {}: {:4.4f}s".format(metric.print_self(), stop - start))
        start = time.perf_counter()

    # 2. Calculation of Laplacian
    L = normalized_laplacian_fullyConnected(S)
    ht.save_hdf5(L, '/home/debu_ch/result/results/Test296/Laplacian_Run1.h5', 'NormalizedFullyConnected')


    if rank == 0:
        stop = time.perf_counter()
        print("Calculation of normalized fully-connected Graph Laplacian : {:4.4f}s".format(stop - start))
        start = time.perf_counter()

    # 3. Eigenvalue and -vector Calculation
    #v0 = ht.ones((L.shape[0],), split=L.split, dtype=ht.float64)/math.sqrt(L.shape[0])
    vr = ht.random.rand(L.shape[0], split=L.split, dtype=ht.float64)
    v0 = vr/ht_norm(vr)
    Vg_norm, Tg_norm = lanczos_ht(L, m, v0)
    ht.save_hdf5(Vg_norm, '/home/debu_ch/result/results/Test296/Lanczos_V_Run1.h5', 'V_NFC')
    ht.save_hdf5(Tg_norm, '/home/debu_ch/result/results/Test296/Lanczos_T_Run1.h5', 'T_NFC')

    if rank == 0:
       stop = time.perf_counter()
       print("Calculation of {} Lanczos iterations : {:4.4f}s".format(m, stop - start))
       start = time.perf_counter()

    # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
    eval, evec = torch.eig(Tg_norm._DNDarray__array, eigenvectors=True)
    # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
    ev = ht.matmul(Vg_norm, ht.factories.array(evec))
    eval_sorted, indices = torch.sort(eval[:, 0], dim=0)
    evec_sorted = ev[:,indices]

    #ht.save_hdf5(evec_sorted, '/home/debu_ch/src/heat/results/Test248/Eigenvectors.h5', 'Evec')

    if rank == 0:
        stop = time.perf_counter()
        print("Calculation of eigenvectors and eigenvalues : {:4.4f}s".format( stop - start))

    # k = 3
    # # 6. cluster Eigenvectors
    # v_star = ev[:,:k].copy()
    # if rank == 0: print(eval_sorted[:k],v_star)
    # kmeans = ht.ml.cluster.KMeans(n_clusters=k)
    # centroids = kmeans.fit(v_star)
    # v_star = v_star.expand_dims(axis=2)
    # distances = ((v_star - centroids) ** 2).sum(axis=1, keepdim=True)
    # matching_centroids = distances.argmin(axis=2, keepdim=True).numpy()
    # tmp = matching_centroids[:, 0, 0]
    # np.savetxt("/home/debu_ch/src/heat/results/Iris/Matching_Centroids.csv",tmp, delimiter=";")


