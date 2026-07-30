'''
Project designed to predict whether a stocks price will go up or down in the next week

It will output the accuracy score of the model and a message saying whether the stock price will rise or not

See the Example Usage section at the bottom of the file for how to use this model
'''

import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def DownloadData(ticker,start,end):
    '''Downloads stock data from Yahoo finance for provided ticker and date range, ideally download at least 5 years of data'''
    data=yf.download(ticker,start=start,end=end)
    return(data)

def RandomForestModel(data):
    '''A Random Forest model trained on stock data to predict whether the stock price will go up or not in the next week'''
    #Features
    data["1DayReturn"]=data["Close"].pct_change(1)
    data["5DayReturn"]=data["Close"].pct_change(5)
    data["10DayReturn"]=data["Close"].pct_change(10)
    data["ExpMovingAvg5"]=data["Close"].ewm(span=5, adjust=False).mean()
    data["ExpMovingAvg20"]=data["Close"].ewm(span=20, adjust=False).mean()
    data["ExpMovingAvgRatio"]=data["ExpMovingAvg5"]/data["ExpMovingAvg20"]
    data["5DayVolatility"]=data["1DayReturn"].rolling(window=5).std()
    data["20DayVolatility"]=data["1DayReturn"].rolling(window=20).std()
    data["VolumeChange"]=data["Volume"].pct_change(1)
    data["VolumeRatio"]=data["Volume"]/(data["Volume"].rolling(window=20).mean())
    data["RSI"]=ta.momentum.RSIIndicator(data["Close"].iloc[:,0], window=14).rsi()
    data["MACD"]=ta.trend.MACD(data["Close"].iloc[:,0]).macd()

    global features
    features=["1DayReturn","5DayReturn",
              "10DayReturn","ExpMovingAvgRatio",
              "5DayVolatility","20DayVolatility",
              "VolumeChange","VolumeRatio",
              "RSI","MACD"]

    #Creating Target
    data["EndofWeekClose"]=data["Close"].shift(-5)
    data["Target"]=(data["EndofWeekClose"]>data["Close"].iloc[:,0]).astype(int)

    #Cleaning Data
    data=data.dropna()

    #Splitting Data into training and testing
    X=data[features]
    Y=data["Target"]

    splitpoint=int(len(data)*0.8)
    XTrain=X[:splitpoint]
    XTest=X[splitpoint:]
    YTrain=Y[:splitpoint]
    YTest=Y[splitpoint:]

    #Training Model
    model=RandomForestClassifier(
        n_estimators=500, 
        max_depth=3,
        min_samples_leaf=2,
        max_features=4,
        )
    model.fit(XTrain,YTrain)

    #Accuracy Test
    predictions=model.predict(XTest)
    accuracy=accuracy_score(YTest,predictions)
    print("Test Accuracy,",accuracy)

    return(model)

def PredictNextWeek(model,currentdata):
    '''Uses model to predict stock price movement for the next week based on current data'''
    #Features
    currentdata["1DayReturn"]=currentdata["Close"].pct_change(1)
    currentdata["5DayReturn"]=currentdata["Close"].pct_change(5)
    currentdata["10DayReturn"]=currentdata["Close"].pct_change(10)
    currentdata["ExpMovingAvg5"]=currentdata["Close"].ewm(span=5, adjust=False).mean()
    currentdata["ExpMovingAvg10"]=currentdata["Close"].ewm(span=10, adjust=False).mean()
    currentdata["ExpMovingAvg20"]=currentdata["Close"].ewm(span=20, adjust=False).mean()
    currentdata["ExpMovingAvgRatio"]=currentdata["ExpMovingAvg5"]/currentdata["ExpMovingAvg20"]
    currentdata["5DayVolatility"]=currentdata["1DayReturn"].rolling(window=5).std()
    currentdata["20DayVolatility"]=currentdata["1DayReturn"].rolling(window=20).std()
    currentdata["VolumeChange"]=currentdata["Volume"].pct_change(1)
    currentdata["VolumeRatio"]=currentdata["Volume"]/(currentdata["Volume"].rolling(window=20).mean())
    currentdata["RSI"]=ta.momentum.RSIIndicator(currentdata["Close"].iloc[:,0],window=14).rsi()
    currentdata["MACD"]=ta.trend.MACD(currentdata["Close"].iloc[:,0]).macd()

    #Clean Data
    currentdata=currentdata.dropna()

    #Predicting using Model
    recent_features=currentdata[features].iloc[-1]
    predictions=model.predict([recent_features])

    if predictions==1:
        print("The stock price is predicted to go up in the next week")
    else:
        print("The stock price is predicted to go down or stay the same in the next week")

'''
Example Usage,

For training data, call the Download Data function with the stock ticker and a start and end date for the training data

For current data call the Download Data function with the stock ticker and a start and end date for the current data.
This should be a date range ending in the current date with a range of at least 20 days.
'''
trainingdata=DownloadData("NVDA","2020-01-01","2026-01-01") 
model=RandomForestModel(trainingdata)
currentdata=DownloadData("NVDA","2026-02-01","2026-07-29") 
PredictNextWeek(model,currentdata)