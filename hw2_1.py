import numpy as np

N_EXPERIMENTS=100000
N_COINS=1000
N_FLIPS=10

v_c_1 = np.zeros(N_EXPERIMENTS)
v_c_rand = np.zeros(N_EXPERIMENTS)
v_c_min = np.ones(N_EXPERIMENTS)

for e in range(N_EXPERIMENTS):
    c_rand = np.random.randint(N_COINS)
    for c in range(1,N_COINS+1):
        n_heads_now = 0
        for f in range(N_FLIPS):
            n_heads_now += (np.random.uniform()>=0.5)
        v_now = n_heads_now/N_FLIPS
        if (v_now < v_c_min[e]):
            v_c_min[e] = v_now
        if (c==1):
            v_c_1[e] = v_now
        elif (c==c_rand):
            v_c_rand[e] = v_now

    if (e>0 and e%1000==0):
        print(e,np.mean(v_c_1[0:e]),np.mean(v_c_rand[0:e]),np.mean(v_c_min[0:e]))

#print(v_c_1)
#print(v_c_rand)
#print(v_c_min)
