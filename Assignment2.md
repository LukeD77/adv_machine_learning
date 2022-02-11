# Assignment 2

### Locally Weighted Regression and Random Forest Regression comparison

Below is the function written for the Locally Weighted Regression 
```python
def kernel_function(xi, x0, kern, tau):
    return kern((xi - x0)/(2*tau)) # Function applies the selected kernal to 

def lowess_regression(x,y,kern,tau):
    n = len(x) # To use for loops across all obsevations
    yhat = np.zeros(n) # Establishing empty yhat vector
    
    w = np.array([kern((x-x[i])/(2*tau)) for i in range (n)]) # Making an array of weights with the kernel function
    # 
    
    for i in range(n):
        weights = w[:,i]
        b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
        A = np.array([[np.sum(weights), np.sum(weights*x)], [np.sum(weights*x), np.sum(weights*x*x)]])
        beta = linalg.solve(A,b)
        yhat[i] = beta[0] + beta[1]*x[i]
    return yhat
```

The Lowess was run on the entire dataset and predictions for the entire dataset were made. The resulting MSE was 10.1, which was calculated using the MSE command from
sklearn.preprocessing. The predicted vs actual values were also plotted.

![actual vs predicted Lowess](https://user-images.githubusercontent.com/67921793/153518113-2a9d8f67-f236-4d84-8859-d13105b5ea4c.png)


Below is the code to intialize, fit, and predict with the Random Forest Regression
```python
model2 = RandomForestRegressor(n_estimators=500, max_depth=3)
model2.fit(xtrain.reshape(-1,1),ytrain)
yhat = model2.predict(xtest.reshape(-1,1))
```
