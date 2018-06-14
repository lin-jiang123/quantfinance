#pip install quandl
import quandl
import datetime
import numpy as np
import pandas as pd
from johansen_test import coint_johansen


 
quandl.ApiConfig.api_key = 'YourQuandlAPI'


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
            collapse='daily',
            order='asc'
            )
 


# Rolling Z-Score for signal generation
def zscore(x, window):
    r = x.rolling(window=window,min_periods=3)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


if __name__ == '__main__':
    
    #Use in-sample data to calculate the Eigen Vector
    
    symbols = ['FED/RXI_N_B_JA','LBMA/GOLD']
    num_assets = len(symbols)    
    FX_data = quandl_stocks(symbols,[1,2],(2010, 6, 10),(2015, 6, 9))
    FX_data.fillna(method='ffill',inplace=True)
    FX_data.iloc[:,0]=1/FX_data.iloc[:,0]
    FX_log_data = pd.DataFrame(index=FX_data.index.copy())
    FX_log_data = np.log(FX_data)
    FX_log_ret_data = pd.DataFrame(index=FX_data.index.copy())
        
    jres=coint_johansen(FX_log_data,0,1) # The inputs are log price levels.        
    weights = jres.evec[0,:]/jres.evec[0,0] # The Eigen Vector is calculated.
    

    #Use out-of-sample data to calculate the stationary series to generate trading signals.
    FX_data = quandl_stocks(symbols,[1,2],(2015, 6, 10),(2018, 6, 10))
    FX_data.fillna(method='ffill',inplace=True)
    FX_data.iloc[:,0]=1/FX_data.iloc[:,0]
    FX_log_data = np.log(FX_data)    
    coint_series = np.matmul(FX_log_data,weights.reshape(2,1))
    coint_series_data = pd.DataFrame(index=FX_data.index.copy())
    coint_series_data['value']=coint_series
    coint_series_data.plot()
    z = zscore(coint_series_data,26)
    z.fillna(0, inplace=True)
    
    signals = pd.DataFrame(index=FX_data.index.copy())
    signals['signal'] = 0.0
    
    #signals[z.iloc[:,0]<-2] = 1.0
    #signals[z.iloc[:,0]>2] = -1.0
    
    for i in range(0,signals.shape[0]):
        
        if i==0 and z.iloc[i,0]<-2:
          signals.iloc[i,0]=1.0
        if i==0 and z.iloc[i,0]>2:
          signals[i]=-1.0
        if i>0 and z.iloc[i,0]<-2 and z.iloc[i-1,0]>-2:
          signals.iloc[i,0]=1.0 #Initialize a buy signal
        if i>0 and z.iloc[i,0]>2 and z.iloc[i-1,0]<2:
          signals.iloc[i,0]=-1.0 #Initialize a sell signal
        if i>0 and z.iloc[i,0]<-1.5 and signals.iloc[i-1,0]==1.0:
          signals.iloc[i,0]=1.0 #Keep the buy signal
        if i>0 and z.iloc[i,0]>1.5 and signals.iloc[i-1,0]==-1.0:
          signals.iloc[i,0]=-1.0 #Keep the sell signal

          
        
    PnL_signals = signals.shift(1) # The reason there is a shift is that once a signal emerges, it will be used to capture the P&L on the next day, since the signal is based on closing prices.
    PnL_signals.fillna(0, inplace=True)
    coint_ret_data = coint_series_data.diff()
    coint_ret_data.fillna(0, inplace=True)
    
    # Cumulative return history since 2015 is stored in the results variable.
    results = np.multiply(PnL_signals,coint_ret_data).cumsum() 
    results.columns=['Cumulative Returns']
    # Output the performance summary results for Dow and Smart Beta
    # results.to_csv(r'C:\Anaconda\Scripts\Lin\Q3.txt', header=True, index=True, sep=' ', mode='w')
    
    results.plot()
    
    
    

    
    



