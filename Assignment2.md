

# Assignment 2 - Locally Weighted Regression and Random Forest Regression comparison

### Locally Weighted Regression (Lowess)
Locally Weighted Regression is a modified version of the standard regression model where the variables are weighted at strengths based upon variable tau to create non-parametric models. The regions of the regression model are calculated by running a kernel across the space and calculating the regression model with the kernel applied over the datapoints, giving weight only to local points in the modeling. This allows for modeling groupings of data instead of the entire dataset. The equation for Lowess Regression is this:

![math1](https://bit.ly/3sALtOq)

When constructing linear regressions, we solve for **beta** assuming that **X^TX** is invertable. When also accounting for weights and then solving for **yhat**, we get the equation:

![math2](https://bit.ly/3sAuNqj)

### Code
Below is the function written for the Locally Weighted Regression 
```python
def tricubic(x):
    return np.where(np.abs(x)>1,0, 70/81*(1-np.abs(x)**3)**3)

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

The Lowess was run on the entire dataset and predictions for the entire dataset were made. Using a tau of 0.0001, the resulting MSE was 10.1, which was calculated using the MSE command from sklearn.preprocessing. The predicted vs actual values were also plotted.

![actual vs predicted Lowess](https://user-images.githubusercontent.com/67921793/153518113-2a9d8f67-f236-4d84-8859-d13105b5ea4c.png)

The clear one-to-one linear trend displays a relatively accurate model.

### Random Forest Regression
Random Forest regression is an ensemble method that utilizes multiple decision trees and averaging the output of all trees to predict y. A depiction of the model workflow is below. Image credit from https://levelup.gitconnected.com/random-forest-regression-209c0f354c84

![image](https://user-images.githubusercontent.com/67921793/153527960-c8ba90d6-18c8-43ce-9adc-9253f0d58a4c.png)


Below is the code to intialize, fit, and predict with the Random Forest Regression
```python
model2 = RandomForestRegressor(n_estimators=500, max_depth=3)
model2.fit(xtrain.reshape(-1,1),ytrain)
yhat = model2.predict(xtest.reshape(-1,1))
```

Using an n_estimator value of 500 and max_depth of 3, the MSE was 34.7. The predicted vs actual values were also plotted.
![predicted vs actual](https://user-images.githubusercontent.com/67921793/153518327-7dc962c8-6237-4de1-813e-2f9d7c8eddf4.png)

The unclear one-to-one linear trend between the predicted and actual vlaues indicates an innacurate model.

### Conclusion

Based upon the MSE and trends in predicted vs actual y values, Locally Weighted Regression outperformed Random Forest Regression by a large margin. Increasing the number of trees or tree depth in the Random Forest model offered no reduction in MSE and moreover no increase in model accuraacy. Utilizing standardized input data resulted in higher MSE for both models and was therefore not used in the final models. 
