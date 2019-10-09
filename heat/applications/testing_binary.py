import heat as ht


if __name__ == "__main__":
    L1 = ht.array([[2,-1,0,0,-1,0],[-1,3, -1,0,-1,0],[0,-1,2,-1,0,0],[0,0,-1,3,-1,-1],[-1,-1,0,-1,3,0],[0,0,0,-1,0,1]], split=0)
    L2 = ht.array([1,1,1,1,1,1])


    print("Rank {} Before L1 = {}\nshape = {}".format(L1.comm.rank, L1, L1.shape))


    print("Rank {} Before L2 = {}\nshape = {}".format(L2.comm.rank, L2, L2.shape))
    L1.comm.Barrier()
    Res = L1 + L2

    print("Rank {} Res = {}".format(Res.comm.rank, Res))

    print("Rank {} After L1 = {}".format(L1.comm.rank, L1))
    print("Rank {} After L2 = {}".format(L2.comm.rank, L2))
