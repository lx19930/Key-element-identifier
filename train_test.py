#import pandas as pd
import numpy as np
#from rg_data3 import load_data
from dataloader import load_data
from sklearn import preprocessing

def getdata(str_score, is_reg = 0, gate= 'avg', droprate=0, thres=0):
    X, y = load_data(str_score, is_reg, gate)
    test_choice = np.random.choice(range(7), 2, replace=False).tolist() +  np.random.choice(range(7,14), 2, replace=False).tolist()
#    test_choice = np.random.choice(range(7), 1, replace=False).tolist() ### only open state remainded
#    test_choice = [2,6]
    #print(test_choice)
    X_train = np.zeros((1,320*320))
    y_train = np.zeros((1,))
    X_test = np.zeros((1,320*320))
    y_test = np.zeros((1,))
    #print(y[1].shape)
    #print(y_test.shape)
    for i in range(14):
        if i in test_choice:
            X_test = np.concatenate((X_test,X[i]),axis=0)
            y_test = np.concatenate((y_test,y[i]),axis=0)
        else:
            X_train = np.concatenate((X_train,X[i]),axis=0)
            y_train = np.concatenate((y_train,y[i]),axis=0)
#        X_test = np.concatenate((X_test,X[i][-50:,:]),axis=0)
#        y_test = np.concatenate((y_test,y[i][-50:]),axis=0)
#        X_train = np.concatenate((X_train,X[i][:-50,:]),axis=0)
#        y_train = np.concatenate((y_train,y[i][:-50]),axis=0)

    X_train = np.delete(X_train,0,0)
    y_train = np.delete(y_train,0,0)
    X_test = np.delete(X_test,0,0)
    y_test = np.delete(y_test,0,0)
   
    m,n = X_train.shape
#    print(m,n)
    mask = np.ones((320,320))
    choice = list(np.random.choice(range(n), int(droprate*n), replace=False))
######## pred
#    choice = choice + [12, 259, 244, 235, 270, 316, 233, 280, 102, 179 ] #+  [246, 266, 198, 286, 240, 217, 58, 40, 215, 242] #+ [130, 287, 9, 178, 241, 281, 114, 128, 126, 131] #+ [211, 120, 106, 284, 214, 290, 115, 279, 197, 15] + [12, 259, 244, 235, 270, 316, 233, 280, 102, 179 ]

######## hbdif
#    choice = choice +  [8, 9, 11, 15, 25, 194, 195, 196, 198, 199] + [200, 57, 61, 229, 66, 231, 236, 255, 92, 272] + [273, 109, 273, 110, 275, 276, 277, 113, 114, 115] + [118, 129, 130, 131, 132, 133, 135, 136, 137]
########M3
#    choice = choice + [5, 6, 7, 10, 12, 13, 14, 19, 22, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 79, 81, 86, 89, 90, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 129, 140, 164, 165, 166, 167, 168, 169, 170, 172, 182, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 249, 250, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 282, 283, 284, 285, 286, 287, 288, 289, 293]
    print(choice)    
    for i in choice:
        mask[:,i] = 0
        mask[i,:] = 0


#    for i in range(160):
#        for j in range(160,320):
#            mask[i,j] = 0

    mask = mask.reshape(1,320*320)
    X_train = X_train * mask
    X_test = X_test * mask


##################   Center
#    X_train = X_train - X_train.mean(0)
#    X_test = X_test - X_test.mean(0)

##################   Normalize Y
#    y_train = y_train/max(y_train)
#    y_test = y_test/max(y_test)
    
###############   indexing
    print('thres=',thres)
    mtr, ntr = X_train.shape
    X_train = X_train.reshape(mtr,320,320)
    mte, nte = X_test.shape
    X_test = X_test.reshape(mte,320,320)
    index = []
    ind1 = []
    ind2 = []
    for i in range(320):
        for j in range(i,320):
            if max(abs(X_train[:,i,j])) > thres:
                index.append((i,j))
                ind1.append(i)
                ind2.append(j)
    X_train = X_train[:,ind1,ind2]
    X_test = X_test[:,ind1,ind2]

#############   Batch norm
#    scaler_train = preprocessing.StandardScaler().fit(X_train)
#    X_train =  scaler_train.transform(X_train)
#    scaler_test = preprocessing.StandardScaler().fit(X_test)
#    X_test =  scaler_test.transform(X_test)
#    
#    y_train = y_train.reshape(-1,1)
#    scaler_trainy = preprocessing.StandardScaler().fit(y_train)
#    y_train =  scaler_trainy.transform(y_train)
#    y_train = y_train.reshape(mtr,)
#    y_test = y_test.reshape(-1,1)
#    scaler_testy = preprocessing.StandardScaler().fit(y_test)
#    y_test =  scaler_testy.transform(y_test)
#    y_test = y_test.reshape(mte,)


############  Layer Norm

    X_train = (X_train - X_train.mean(axis=1).reshape(mtr,1))/X_train.std(axis=1).reshape(mtr,1)
    X_test = (X_test - X_test.mean(axis=1).reshape(mte,1))/X_test.std(axis=1).reshape(mte,1) 
 
    y_train = y_train.reshape(-1,1)
    scaler_trainy = preprocessing.StandardScaler().fit(y_train)
    y_train =  scaler_trainy.transform(y_train)
    y_train = y_train.reshape(mtr,)
    y_test = y_test.reshape(-1,1)
    scaler_testy = preprocessing.StandardScaler().fit(y_test)
    y_test =  scaler_testy.transform(y_test)
    y_test = y_test.reshape(mte,)






###### Use only 400 O state data
#    y_test = y_test[400:]
#    X_test = X_test[400:,:]
    print('X_train.shape = ',X_train.shape)
    print('X_test.shape = ',X_test.shape)
    print('y_train.shape = ',y_train.shape)
    print('y_test.shape = ',y_test.shape)
    return X_train, y_train, X_test, y_test, test_choice, index

#print(X_temp[0])
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#getdata('vdw')
