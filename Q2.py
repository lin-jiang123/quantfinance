#pip install quandl
import quandl
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
quandl.ApiConfig.api_key = 'YourQuandlAPI'


def quandl_stocks(symbols, start_date=(2010, 1, 1), end_date=None):
    """
    symbol is a string representing a stock symbol, e.g. 'AAPL'
 
    start_date and end_date are tuples of integers representing the year, month,
    and day
 
    end_date defaults to the current date when None
    """
 
    query_list = ['WIKI' + '/' + symbols[k] + '.' + str(4) for k in range(0,len(symbols))]
    # str(4) refers to closing price
    """query_list = ['WIKI' + '/' + symbol + '.' + str(k) for k in range(1, 13)]"""
 
    start_date = datetime.date(*start_date)
 
    if end_date:
        end_date = datetime.date(*end_date)
    else:
        end_date = datetime.date.today()
 
    return quandl.get(query_list, 
            returns='pandas', 
            start_date=start_date,
            end_date=end_date,
            collapse='weekly',
            order='asc'
            )
 
# Sharpe Ratio 
def sharpe(returns, rf, scaler):
    volatility = returns.std() * np.sqrt(scaler) 
    sharpe_ratio = (returns.mean() - rf)*scaler/ volatility
    return sharpe_ratio
    

def max_drawdown(cum_returns):
    
    i = np.argmax(np.maximum.accumulate(cum_returns) - cum_returns) # end of the period
    j = np.argmax(cum_returns[:i]) # start of period
    

    return cum_returns[i]-cum_returns[j]


    
if __name__ == '__main__':
    
    symbols = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DWDP','XOM','GE','GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','TRV','UTX','UNH','VZ','V','WMT']
    dow_data = quandl_stocks(symbols)
    num_stocks = len(symbols)
    dow_data['DOW'] = dow_data.sum(axis=1)
    
    for i in range(0,num_stocks):
        
     dow_data['log_ret'+str(i+1)] = np.log(dow_data.iloc[:,i]) - np.log(dow_data.iloc[:,i].shift(1))

    dow_data['log_ret_DOW'] = np.log(dow_data.DOW) - np.log(dow_data.DOW.shift(1))
    dow_data['log_ret_ew_DOW']= dow_data.iloc[:,(num_stocks+1):(num_stocks*2+1)].mean(axis=1)
    
    
    dow_data['cum_ret_DOW'] = dow_data['log_ret_DOW'].cumsum()
    dow_data['cum_ret_ew_DOW'] = dow_data['log_ret_ew_DOW'].cumsum()
    
    dow_data.fillna(0, inplace=True)
    
    
    sr_dow_1 = sharpe(dow_data['log_ret_DOW'], 0, 52)
    sr_dow_2 = sharpe(dow_data['log_ret_ew_DOW'], 0, 52)
    
    md_dow_1 = max_drawdown(dow_data['cum_ret_DOW'])
    md_dow_2 = max_drawdown(dow_data['cum_ret_ew_DOW'])
    
    performance_data = [['Price Weighted Dow Index',sr_dow_1,md_dow_1],['Equally Weighted Dow Index',sr_dow_2,md_dow_2]]
    
    
    # Cumulative return history since 2015 is stored in the results variable.
    results = pd.DataFrame(performance_data,columns=['Index Type','Sharpe Ratio','Max Drawdown'])
    results.set_index('Index Type', inplace=True)
    
    # Output the performance summary results for Dow and Smart Beta
    # results.to_csv(r'C:\Anaconda\Scripts\Lin\Q2.txt', header=True, index=True, sep=' ', mode='w')
  
    
    plt.plot_date(x=dow_data.index.get_values(), y=dow_data['cum_ret_DOW'], fmt="r-")
    plt.plot_date(x=dow_data.index.get_values(), y=dow_data['cum_ret_ew_DOW'], fmt="b-")
    plt.title("Dow vs. Equited Weighted Dow (Smart Beta) since 2010")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()
    




