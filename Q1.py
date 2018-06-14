import quandl
import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests as gc
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math

 
quandl.ApiConfig.api_key = 'iMjgDUbUjVyCLx4iDLk5'


def quandl_stocks(symbols, idx, start_date, end_date):
    """
    symbol is a string representing a stock symbol, e.g. 'AAPL'
 
    start_date and end_date are tuples of integers representing the year, month,
    and day
 
    end_date defaults to the current date when None
    """
 
    query_list = [symbols[k] + '.' + str(idx[k]) for k in range(0,len(symbols))]
 
    start_date = datetime.date(*start_date)
 
    if end_date:
        end_date = datetime.date(*end_date)
    else:
        end_date = datetime.date.today()
 
    return quandl.get(query_list, 
            returns='pandas', 
            start_date=start_date,
            end_date=end_date,
            collapse='monthly',
            order='asc'
            )
 

def optimal_lag(granger_test_result):
    optimal_lag = -1
    pval_test = 1
    for key in granger_test_result.keys():
     _pval_test_ = granger_test_result[key][0]['params_ftest'][1]
     if _pval_test_ < pval_test:
        pval_test = _pval_test_
        optimal_lag = key
    return optimal_lag


if __name__ == '__main__':
    
    
    '''
    BLSI/CUUR0000SA0: US Consumer Price Index
    BKRHUGHES/RIGS_BY_STATE_TOTALUS_TOTAL: Baker Hughes total rig counts
    ISM/MAN_PRICES: Price sentiment survey information collected by ISM
    '''
   
    symbols = ['BLSI/CUUR0000SA0','BKRHUGHES/RIGS_BY_STATE_TOTALUS_TOTAL','ISM/MAN_PRICES']
    num_assets = len(symbols)
    
       
    level_data = quandl_stocks(symbols,[1,1,5],(2000, 2, 7),(2018, 4, 30))
    split_idx = math.floor(len(level_data.index)*0.75)
    
    change_data = level_data.pct_change()
    change_data.fillna(0,inplace=True)
    
    lvl_in_sample = level_data.iloc[:split_idx,:]
    
    lvl_out_of_sample = level_data.iloc[(split_idx):len(change_data.index),:].shift(1)
    lvl_out_of_sample = lvl_out_of_sample.iloc[1:]
    
    chg_in_sample = change_data.iloc[:split_idx,:]
    chg_out_of_sample = change_data.iloc[(split_idx+1):len(change_data.index),:]
    
    granger_rigs = gc(chg_in_sample.iloc[:,[0,1]], maxlag = 12)
    granger_consumer = gc(chg_in_sample.iloc[:,[0,2]], maxlag = 12)
    
    rigs_lag = optimal_lag(granger_rigs)
    consumer_lag = optimal_lag(granger_consumer)
    
    # Creating training set and test set
    
    X_train = pd.DataFrame(index=chg_in_sample.index.copy())
    X_test = pd.DataFrame(index=chg_out_of_sample.index.copy())
    
    for i in range(0,np.max([rigs_lag,consumer_lag])):
        
     X_train['CPI - '+str(i+1) + ' months ago'] = chg_in_sample.iloc[:,0].shift(i+1)
     X_test['CPI - '+str(i+1) + ' months ago'] = chg_out_of_sample.iloc[:,0].shift(i+1)
    
    for i in range(0,rigs_lag):
        
     X_train['Rigs - '+str(i+1) + ' months ago'] = chg_in_sample.iloc[:,1].shift(i+1)
     X_test['Rigs - '+str(i+1) + ' months ago'] = chg_out_of_sample.iloc[:,1].shift(i+1)
    
    for i in range(0,consumer_lag):
        
     X_train['Survey - '+str(i+1) + ' months ago'] = chg_in_sample.iloc[:,2].shift(i+1)
     X_test['Survey - '+str(i+1) + ' months ago'] = chg_out_of_sample.iloc[:,2].shift(i+1)
    
    X_train_adj = X_train.iloc[np.max([rigs_lag,consumer_lag]):]
    y_train_adj = chg_in_sample.iloc[np.max([rigs_lag,consumer_lag]):,0]
    
    X_test_adj = X_test.iloc[np.max([rigs_lag,consumer_lag]):]
    y_test_adj = chg_out_of_sample.iloc[np.max([rigs_lag,consumer_lag]):,0]
    lvl_out_of_sample_adj = lvl_out_of_sample.iloc[np.max([rigs_lag,consumer_lag]):,0]
    
    lm = linear_model.LinearRegression()
    lm.fit(X_train_adj, y_train_adj)
    yhat = lm.predict(X_train_adj)
    print(r2_score(y_train_adj, yhat))

        
    ypred = lm.predict(X_test_adj)
    print(r2_score(y_test_adj, ypred))

    
    results = pd.DataFrame(index=X_test_adj.index.copy())
    results['Forecast CPI'] =  (ypred + 1) * lvl_out_of_sample_adj
    results['Actual CPI'] =  (y_test_adj + 1) * lvl_out_of_sample_adj
    
    # Output the forcast and actual CPI levels
    # results.to_csv(r'C:\Anaconda\Scripts\Lin\Q1.txt', header=True, index=True, sep=' ', mode='w')
    
    plt.plot_date(x=results.index.get_values(), y=results['Forecast CPI'], fmt="r-")
    plt.plot_date(x=results.index.get_values(), y=results['Actual CPI'], fmt="b-")
    plt.title("Forecast CPI Index (Red) vs. Actual CPI Index (Blue)")
    plt.ylabel("Index")
    plt.grid(True)
    plt.show()
    
    
    
    

    
    



