import pandas as pd
import numpy as np

X = []
y = []

def load_data(str_score, is_reg, gate):

    if str_score=='vdw':
        file_1 = 'N1_vdw.dat'
        file_2 = 'N2_vdw.dat'
        file_12 = 'N12_vdw.dat'
        close_1 = 'N1_vdw_close.dat'
        close_2 = 'N2_vdw_close.dat'
        close_12 = 'N12_vdw_close.dat'
    elif str_score=='ele':
        file_1 = 'N1_ele.dat'
        file_2 = 'N2_ele.dat'
        file_12 = 'N12_ele.dat'
        close_1 = 'N1_ele_close.dat'
        close_2 = 'N2_ele_close.dat'
        close_12 = 'N12_ele_close.dat'
    elif str_score=='hbpair':
        file_1 = 'hbpair_N1.dat'
        file_2 = 'hbpair_N2.dat'
        file_12 = 'hbpair_N12.dat'
        close_1 = 'hbpair_N1_close.dat'
        close_2 = 'hbpair_N2_close.dat'
        close_12 = 'hbpair_N12_close.dat'
    elif str_score=='total':
        file_1 = 'N1_total.dat'
        file_2 = 'N2_total.dat'
        file_12 = 'N12_total.dat'
        close_1 = 'N1_total_close.dat'
        close_2 = 'N2_total_close.dat'
        close_12 = 'N12_total_close.dat'

    file_y = 'rg_7traj.dat'
    close_y = 'rg_close.dat'
    file_y2 = 'rg_m_open.dat'
    close_y2 = 'rg_m_close.dat'
 
    A = np.loadtxt(file_1,delimiter = ',')
    A2 = np.loadtxt(close_1,delimiter = ',')
    A = np.concatenate((A, A2), axis=0)
    print('Ashape',A.shape)
    m,n = A.shape
    ndata = m
    A = A.reshape(ndata,160,160)
    A = A + np.transpose(A,axes = (0,2,1))

    B = np.loadtxt(file_2,delimiter = ',')
    B2 = np.loadtxt(close_1,delimiter = ',')
    B = np.concatenate((B, B2), axis=0)
    print('Bshape',B.shape)
    B = B.reshape(ndata,160,160)
    B = B + np.transpose(B,axes = (0,2,1))


    C = np.loadtxt(file_12,delimiter = ',')
    C2 = np.loadtxt(close_12,delimiter = ',')
    C = np.concatenate((C, C2), axis=0)
    print('Cshape',C.shape)
    C = C.reshape(ndata,160,160)


    D = np.loadtxt(file_y,delimiter = ',')
    DD = np.loadtxt(file_y2)  ##avg = 10 smallest rg in open state
    DD = DD.reshape(5600,1)
    D = np.concatenate((D, DD), axis=1)
    m1, n1 = D.shape
    d1_avg = D.mean(0)
    D2 = np.loadtxt(close_y,delimiter = ' ')
    DD2 = np.loadtxt(close_y2)    ##avg = 10 smallest rg in close state
    DD2 = DD2.reshape(2886,1)
    D2 = np.concatenate((D2, DD2), axis=1)
    m2, n2 = D2.shape
    d2_avg = D2.mean(0)
    D = np.concatenate((D, D2), axis=0)
    print('d1_avg',d1_avg)
    print('d2_avg',d2_avg)
    print('Dshape',D.shape)

    data1 = np.concatenate((A, C), axis=2)
    print(data1.shape)
    data2 = np.concatenate((np.transpose(C,axes = (0,2,1)), B), axis=2)
    print(data2.shape)
    data = np.concatenate((data1, data2), axis=1)
    print(data.shape)
    
    if gate == 'avg': 
        tgt = (D[:,0] + D[:,3])/2
        thre = (d1_avg[0] + d1_avg[3] + d2_avg[0] + d2_avg[3])/4
        thre = 7.3
    elif gate == 'upper':
        tgt = D[:,0]
        thre = (d1_avg[1] + d2_avg[1])/2
    elif gate == 'lower':
        tgt = D[:,3]
        thre = (d1_avg[3] + d2_avg[3])/2
        thre = 6.8
    elif gate == 'pore':
        tgt = D[:,6]
        thre = (d1_avg[6] + d2_avg[6])/2
        thre = 4.7



    print(thre)
    if is_reg == 0:
        tgt = 1*(tgt>thre)
    ndata = data.shape[0]
    data = data.reshape(ndata,320*320) 
    
    for i in range(7):
        X.append(data[i*800:(i+1)*800,:])
        y.append(tgt[i*800:(i+1)*800])

    close_length = [430, 563, 491, 381, 301, 320, 400]
    tail = 5600
    for i in range(len(close_length)):
        head = tail
        tail = head + close_length[i]
        X.append(data[head:tail,:])
        y.append(tgt[head:tail])
  

    return X, y
#load_data('vdw', 0, 'upper')
