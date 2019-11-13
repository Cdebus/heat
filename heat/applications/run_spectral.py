# # 2. Calculation of Laplacian
# L = normalized_laplacian_fullyConnected(S)
# ht.save_hdf5(L, '/home/debu_ch/result/results/Test296/Laplacian_Run1.h5', 'NormalizedFullyConnected')
#
#
# if rank == 0:
#     stop = time.perf_counter()
#     print("Calculation of normalized fully-connected Graph Laplacian : {:4.4f}s".format(stop - start))
#     start = time.perf_counter()
#
# # 3. Eigenvalue and -vector Calculation
# #v0 = ht.ones((L.shape[0],), split=L.split, dtype=ht.float64)/math.sqrt(L.shape[0])
# vr = ht.random.rand(L.shape[0], split=L.split, dtype=ht.float64)
# v0 = vr/ht_norm(vr)
# Vg_norm, Tg_norm = lanczos_ht(L, m, v0)
# ht.save_hdf5(Vg_norm, '/home/debu_ch/result/results/Test296/Lanczos_V_Run1.h5', 'V_NFC')
# ht.save_hdf5(Tg_norm, '/home/debu_ch/result/results/Test296/Lanczos_T_Run1.h5', 'T_NFC')
#
# if rank == 0:
#    stop = time.perf_counter()
#    print("Calculation of {} Lanczos iterations : {:4.4f}s".format(m, stop - start))
#    start = time.perf_counter()
#
# # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
# eval, evec = torch.eig(Tg_norm._DNDarray__array, eigenvectors=True)
# # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
# ev = ht.matmul(Vg_norm, ht.factories.array(evec))
# eval_sorted, indices = torch.sort(eval[:, 0], dim=0)
# evec_sorted = ev[:,indices]
#
# #ht.save_hdf5(evec_sorted, '/home/debu_ch/src/heat/results/Test248/Eigenvectors.h5', 'Evec')
#
# if rank == 0:
#     stop = time.perf_counter()
#     print("Calculation of eigenvectors and eigenvalues : {:4.4f}s".format( stop - start))

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


