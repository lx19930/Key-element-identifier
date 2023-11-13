import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc
import xgboost as xgb
import shap
import pandas as pd
from math import sqrt, isnan
from sklearn.linear_model import LinearRegression, RidgeCV, LassoLarsCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
from scipy.stats.stats import pearsonr, pointbiserialr, spearmanr
from train_test import getdata
from scipy import stats
from sklearn.metrics import mean_squared_error

def xgb_fit(X_train, y_train, X_test, y_test, f, md, nt, stg, is_reg):
    colsamp = 0.05
    subsam = 1
    print("XGBRegressor/XGBClassifier:",f, md, nt, colsamp)
    if is_reg==1:
        mod = xgb.XGBRegressor (n_estimators=nt, reg_alpha=f, reg_lambda=f, max_depth=md,  scale_pos_weight=0.4, colsample_bytree=1, colsample_bylevel=colsamp, colsample_bynode=colsamp, eta=0.04, subsample=subsam) # L1, L2 regularization weights: reg_alpha=0, reg_lambda=1
    else:
        mod = xgb.XGBClassifier (n_estimators=nt, reg_alpha=f, reg_lambda=f, max_depth=md, scale_pos_weight=0.4, colsample_bytree=1, colsample_bylevel=colsamp, colsample_bynode=colsamp, eta=0.04, subsample=subsam)
    print(mod)
    eval_set = [(X_test,y_test)]
    model = mod.fit(X_train, y_train, eval_set=eval_set, verbose=True)#, early_stopping_rounds=20)
    results = model.evals_result()
    print(results)
    preds = model.predict(X_train)
    test = y_train
    if is_reg == 0:
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('xgb_train')
        print(A)
        preds = model.predict(X_test)
        test = y_test
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('xgb_test')
        print(A)
        print(A.shape)
    
        r2 = mod.score(X_test, y_test)    
        print("r2/acc="+str(r2))
    
        F1 = 2*A[0][0]/(2*A[0][0] + A[0][1] + A[1][0])
        print("F1=",F1)
    elif is_reg == 1:
        print('xgb_train')
        r2 = mod.score(X_train, y_train)
        print("r2/acc="+str(r2))
        preds = model.predict(X_train)
        print('mse=' + str(mean_squared_error(preds, y_train)))
        print('xgb_test')
        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))
        preds = model.predict(X_test)
        print('mse=' + str(mean_squared_error(preds, y_test)))

    grad = model.feature_importances_
    shap_values_avg, shap_values_abs = call_shap(X_train, model)
    return grad, shap_values_avg, shap_values_abs

def rf_fit(X_train, y_train, X_test, y_test, stg, nt, md, is_reg):
    print("RandomForestRegressor/RandomForestClassifier:",nt,md)

    if is_reg==1:
        mod = RandomForestRegressor(n_jobs=-1, max_depth=md, n_estimators=nt, oob_score=True, max_features="sqrt")
    else:
        mod = RandomForestClassifier(n_jobs=-1, max_depth=md, n_estimators=nt, oob_score=True, max_features="sqrt")

    print(mod)
    model = mod.fit(X_train, y_train)
    preds = model.predict(X_train)
    test = y_train
    if is_reg == 0:
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('rf_train')
        print(A)
        preds = model.predict(X_test)
        test = y_test
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('rf_test')
        print(A)
        print(A.shape)

        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))

        F1 = 2*A[0][0]/(2*A[0][0] + A[0][1] + A[1][0])
        print("F1=",F1)
    elif is_reg == 1:
        print('rf_train')
        r2 = mod.score(X_train, y_train)
        print("r2/acc="+str(r2))
        preds = model.predict(X_train)
        print('mse=' + str(mean_squared_error(preds, y_train)))
        print('rf_test')
        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))
        preds = model.predict(X_test)
        print('mse=' + str(mean_squared_error(preds, y_test)))

    grad = model.feature_importances_
    print('grad')
    print(grad)
    print(grad.shape)
    if is_reg==1:
        shap_values_avg, shap_values_abs = call_shap(X_train, model)
    else:
        shap_values_avg, shap_values_abs = call_shap(X_train, model, check_additivity=False)
    return grad, shap_values_avg, shap_values_abs

def svm_fit(X_train, y_train, X_test, y_test, stg, c, is_reg):
    assert is_reg == 1
    print("SVR:C="+str(c))
#    mod = SVR(tol=1e-5, C=c, kernel='rbf')
    mod = SVR(kernel='rbf')
    print(mod)
    model = mod.fit(X_train, y_train)
    print('svm_train')
    r2 = mod.score(X_train, y_train)
    print("r2/acc="+str(r2))
    preds = model.predict(X_train)
    print('mse=' + str(mean_squared_error(preds, y_train)))
    print('svm_test')
    r2 = mod.score(X_test, y_test)
    print("r2/acc="+str(r2))
    preds = model.predict(X_test)
    print('mse=' + str(mean_squared_error(preds, y_test)))
    grad = model.dual_coef_
    shap_values_avg, shap_values_abs = call_shap(X_train, model)
#    shap_values_avg, shap_values_abs = grad, grad
    return grad, shap_values_avg, shap_values_abs


def ada_fit(X_train, y_train, X_test, y_test, stg, nt, is_reg):
    if  is_reg == 1:
        print("AdaBoostRegressor:"+str(nt))
        mod = AdaBoostRegressor(random_state=0,n_estimators=nt)
    elif is_reg == 0:
        mod = AdaBoostClassifier(random_state=0,n_estimators=nt)
    
    model = mod.fit(X_train, y_train)
    preds = model.predict(X_train)
    test = y_train
    if is_reg == 0:
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('ab_train')
        print(A)
        preds = model.predict(X_test)
        test = y_test
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('ab_test')
        print(A)
        print(A.shape)

        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))

        F1 = 2*A[0][0]/(2*A[0][0] + A[0][1] + A[1][0])
        print("F1=",F1)
    elif is_reg == 1:
        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))
        preds = model.predict(X_test)
        print('mse=' + str(mean_squared_error(preds, y_test)))
    grad = model.feature_importances_
    shap_values_avg, shap_values_abs = call_shap(X_train, model)
    return grad, shap_values_avg, shap_values_abs

def lr_fit(X_train, y_train, X_test, y_test, stg, is_reg):
    assert is_reg == 1
#    mod = LinearRegression()
    mod = Lasso(alpha=0.01)
    print(mod)
    model = mod.fit(X_train, y_train)
    print('lr_train')
    r2 = mod.score(X_train, y_train)
    print("r2/acc="+str(r2))
    preds = model.predict(X_train)
    print('mse=' + str(mean_squared_error(preds, y_train)))
    print('lr_test')
    r2 = mod.score(X_test, y_test)
    print("r2/acc="+str(r2))
    preds = model.predict(X_test)
    print('mse=' + str(mean_squared_error(preds, y_test)))
    grad = model.coef_
    shap_values_avg, shap_values_abs = call_shap(X_train, model)
#    shap_values_avg, shap_values_abs = grad, grad
    return grad, shap_values_avg, shap_values_abs



def gb_fit(X_train, y_train, X_test, y_test, f, md, nt, stg, is_reg):
    print("GradientBoostingRegressor/GradientBoostingClassifier: n_estimators,max_depth=",nt,md)
    #reg = GradientBoostingRegressor(random_state=0,n_estimators=nt,max_depth=md,max_features="sqrt")
    subsam = 1
    if is_reg==1:
        mod = GradientBoostingRegressor(subsample=subsam, learning_rate = 0.04, n_estimators=nt,max_features="sqrt")
    else:
        mod = GradientBoostingClassifier(subsample=subsam, learning_rate = 0.04, n_estimators=nt,max_features="sqrt")

    print(mod)

    model = mod.fit(X_train, y_train)
    preds = model.predict(X_train)
    test = y_train
    if is_reg == 0:
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('gb_train')
        print(A)
        preds = model.predict(X_test)
        test = y_test
        A = pd.crosstab(test, preds, rownames=['Actual State'], colnames=['Predicted State'])
        print('gb_test')
        print(A)
        print(A.shape)

        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))

        F1 = 2*A[0][0]/(2*A[0][0] + A[0][1] + A[1][0])
        print("F1=",F1)
    elif is_reg == 1:
        print('gb_train')
        r2 = mod.score(X_train, y_train)
        print("r2/acc="+str(r2))
        preds = model.predict(X_train)
        print('mse=' + str(mean_squared_error(preds, y_train)))
        print('gb_test')
        r2 = mod.score(X_test, y_test)
        print("r2/acc="+str(r2))
        preds = model.predict(X_test)
        print('mse=' + str(mean_squared_error(preds, y_test)))
    grad = model.feature_importances_
    shap_values_avg, shap_values_abs = call_shap(X_train, model)
    return grad, shap_values_avg, shap_values_abs



def call_shap(features,model,check_additivity=True):
    print("call_shap:",use_shap_kernel)
    def call_mod(X):
        return model.predict(X)
    if use_shap_kernel==0: # using TreeExplainer: fast and reproducible
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features, check_additivity=check_additivity) 
    elif use_shap_kernel==1:
#        X_train_summary = shap.kmeans(features, 3)
#        explainer = shap.KernelExplainer(call_mod, X_train_summary)
        explainer = shap.LinearExplainer(model, features)
        shap_values = explainer.shap_values(features)

    if (check_additivity==False):
        shap_values = shap_values[0]
    print(shap_values)
    shap_values_avg1 = shap_values.mean(0)
    shap_values_abs1 = np.abs(shap_values).mean(0)  
    
    return shap_values_avg1, shap_values_abs1 




def grades_to_residue(grad, stg):
    if stg == 'max':
        max_score = grad.max(0)
        score1 = max_score[:160]
        score2 = max_score[160:]
    elif stg == 'sum':
        sum_score = grad.sum(0)
        score1 = sum_score[:160]
        score2 = sum_score[160:]

    return score1, score2


def output_score_NMDA(grad0, grad1, grad2, stg):
    score1_0, score2_0 = grades_to_residue(grad0,stg)
    score1_1, score2_1 = grades_to_residue(grad1,stg)
    score1_2, score2_2 = grades_to_residue(grad2,stg)

    for i in range(160): # N1 residues
        i1=i
        if i1<120:
            resi = 544+i1
        else:
            resi = 799+i1-120
        #print(f'R1_{resi}_{ind} g={score1_0[i]} {score1_1[i]} {score1_2[i]}', flush=True)
        print(f'R1_{resi} g= {score1_0[i]} {score1_1[i]} {score1_2[i]} ', flush=True)

    for i in range(160): # N2 residues
        i1=i
        if i1<120:
            resi = 539+i1
        else:
            resi = 803+i1-120
        #print(f'R2_{resi}_{ind} g={score2_0[i]} {score2_1[i]} {score2_2[i]}', flush=True)
        print(f'R2_{resi} g= {score2_0[i]} {score2_1[i]} {score2_2[i]} ', flush=True)

for repeat in range(1):
    ind = 0
    str_score = sys.argv[1]
    is_reg = sys.argv[2]
    is_reg = int(is_reg)
    gate = sys.argv[4]
    thres = float(sys.argv[8])
    X_train, y_train, X_test, y_test, test_choice, index = getdata(str_score, is_reg, gate, droprate= 0, thres= thres)
    print('testset',test_choice)
    
    
    
    use_shap_kernel = 0    
    model_str = str(sys.argv[3])

    if model_str=='xgb':
        #f = float(sys.argv[3])
        #md = int(sys.argv[4])
        nt = int(sys.argv[5])
        f = float(sys.argv[6])
        stg = sys.argv[7]
        md = 3
        grad, shap_avg, shap_abs = xgb_fit(X_train, y_train, X_test, y_test, f, md, nt, stg, is_reg)

    if model_str=='rf':
        #f = float(sys.argv[3])
        #md = int(sys.argv[4])
        nt = int(sys.argv[5])
        f = float(sys.argv[6])
        stg = sys.argv[7]
        md = 8
        grad, shap_avg, shap_abs = rf_fit(X_train, y_train, X_test, y_test, stg, nt, md, is_reg)

    if model_str=='svm':
        use_shap_kernel = 1
        c = float(sys.argv[5])
        stg = sys.argv[6]
        grad, shap_avg, shap_abs = svm_fit(X_train, y_train, X_test, y_test, stg, c, is_reg)

    if model_str=='ab':
        nt = int(sys.argv[5])
        f = float(sys.argv[6])
        stg = sys.argv[7]
        md = 3
        grad, shap_avg, shap_abs = ada_fit(X_train, y_train, X_test, y_test, stg, nt, is_reg)

    if model_str=='lr':
        nt = int(sys.argv[5])
        f = float(sys.argv[6])
        stg = sys.argv[7]
        md = 3
        use_shap_kernel = 1
        grad, shap_avg, shap_abs = lr_fit(X_train, y_train, X_test, y_test, stg, is_reg)

    if model_str=='gb':
        nt = int(sys.argv[5])
        f = float(sys.argv[6])
        stg = sys.argv[7]
        md = 2
        grad, shap_avg, shap_abs = gb_fit(X_train, y_train, X_test, y_test, f, md, nt, stg, is_reg)

######## PROJ
    count = 0
    new_grad = np.zeros((320,320))
    for (i,j) in index:
        new_grad[i,j] = grad[count]
        count += 1
    grad = new_grad.reshape(320*320)

    count = 0
    new_avg = np.zeros((320,320))
    for (i,j) in index:
        new_avg[i,j] = shap_avg[count]
        count += 1
    shap_avg = new_avg.reshape(320*320)

    count = 0
    new_abs = np.zeros((320,320))
    for (i,j) in index:
        new_abs[i,j] = shap_abs[count]
        count += 1
    shap_abs = new_abs.reshape(320*320)



 
    print('grad_shape',grad.shape)
    print('shap_avg_shape',shap_avg.shape)
    print('shap_abs_shape',shap_abs.shape)
    grad = grad.reshape(320,320)
    shap_avg = shap_avg.reshape(320,320)
    shap_abs = shap_abs.reshape(320,320)
    grad = grad + np.transpose(grad,axes = (1,0))
    shap_avg = shap_avg + np.transpose(shap_avg,axes = (1,0))
    shap_abs = shap_abs + np.transpose(shap_abs,axes = (1,0))
    print('grad\n')
    print(grad)
    print('shap_avg\n')
    print(shap_avg)
    print('shap_abs\n')
    print(shap_abs)
    out_grad = 'grad_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_' + sys.argv[4] + '_ln_open'
    np.save(out_grad,shap_abs)
    output_score_NMDA(grad, shap_avg, shap_abs, stg)
