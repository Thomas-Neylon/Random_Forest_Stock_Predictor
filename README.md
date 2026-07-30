# Random_Forest_Stock_Predictor
A project designed to predict whether a stock ends higher or lower/unchanged at the end of the week by using the Random Forest machine learning algorithm.

The model is provided with financial data from yahoo finance. It then uses multiple features such as day returns, exponential moving averages and RSI to train a random forest model to predict whether a given stock will rise or not by the end of next week. 

After testing, I refined the features used in the model as well as its hyperparameters to get an accuracy of about 53%-58% for major stocks such as NVDA, AMZN and MSFT.

The example usage section at the bottom of the program gives guidance on how it can be used.

This program requires yahoo finance, numpy, pandas, ta and sklearn to be downloaded.

