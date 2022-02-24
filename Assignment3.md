# Assignment 3

### Extreme gradient boost

Extreme Gradient Boost (XGBoost) is a method of machine learning that uses decision trees creating splits of the data and examining based upon calculations including the residuals for making new decision tree leaves. The calculation used is called the gain function; the equation of which is below.

![gain function](https://user-images.githubusercontent.com/67921793/155613003-5e788a89-61ae-4e02-a7ad-373cd380c50f.png)

- **G<sub>L</sub>** is the sum of residuals on the left split
- **G<sub>R</sub>** is the sum of residuals on the right split
- **H<sub>R</sub>** is the number of residuals on the right split
- **H<sub>L</sub>** is the number of residuals on the left split
- **λ** is the regularization parameter which makes the individual observations less influencial to model predictions
- **γ** is the required loss to split a node into two more leaves on the model

After all splits are made and gain functions are evaluated, the split yielding the largest gain function. When all splits yield negative gain values, then the tree reaches the end. This tree then predicts on the train data and new residuals are created to train another tree. This process is repeated until the number of estimators is reached. Then the sum of each decision tree times the learning rate because the final prediction of the test data.

### Code for running models

**Lowess Regression**
```python
def lowess_reg(X, y, xnew, kernel, tau, intercept):
    n = len(X)
    yest = np.zeros(n)
    
    if len(y.shape)==1:
        y = y.reshape(-1,1)
        
    if len(X.shape)==1:
        X = X.reshape(-1,1)
        
    if intercept:
        x1 = np.column_stack([np.ones((len(X),1)),X])
    else:
        x1 = X
        
    weight_array = np.array([kernel((X-X[i])/(2*tau)) for i in range(n)])
    
    for i in range(n):
        weight = np.diag(weight_array[:,i])
        b = np.transpose(x1).dot(weight).dot(y)
        A = np.transpose(x1).dot(weight).dot(x1)
        beta, res,rnk,s = lstsq(A,b)
        yest[i] = np.dot(x1[i],beta)
        
    if X.shape[1]==1:
        f = interp1d(X.flatten(), yest,fill_value='extrapolate')
    else:
        f = LinearNDInterpolator(X, yest)
    output = f(xnew)
    
    if sum(np.isnan(output))>0:
        g = NearestNDInterpolator(X,y.ravel())
        output[np.isnan(output)] = g(xnew[np.isnan(output)])
        
    return output
```

**Boosted Lowess Regression**
```python
def boosted_lowess_reg(X,y,xnew,kernel,tau,intercept):
    Fx = lowess_reg(X,y,X,kernel,tau,intercept)
    new_y = y - Fx
    
    model = RandomForestRegressor(n_estimators=100, max_depth=2)
    model.fit(X,new_y)
    output = model.predict(xnew) + lowess_reg(X,y,xnew,kernel,tau,intercept)
    
    return output
```

**Random Forest**
```python
        model_rf = RandomForestRegressor(n_estimators = 300, max_depth = 7)
        model_rf.fit(xtrain, ytrain)
        yhat_rf = model_rf.predict(xtest)
```

**Extreme Gradient Boosting**
```python
        model_boostrf = xgb.XGBRegressor(objective='reg:squarederror',
                                         n_estimators=300,reg_lambda=20,alpha=1,gamma=10,max_depth=7)
        model_boostrf.fit(xtrain,ytrain)
        yhat_boostrf = model_boostrf.predict(xtest)
```

 The models used the Boston Housing Price data with crime rate, number of rooms, and distance to predict the housing price. After running the models through 5 split kfold 50 times, the following MSE and MAE values were produced. 
 
| Model | MSE    | MAE
| :---:   | :---: | :---: |
| Lowess Reg | 28.65     | 3.29
| Boosted Lowess Reg | 26.20 | 3.31
| Random Forest | 26.19 | 3.20
| XGB | 22.08 | 3.11
 
 - present MSE and MAE for code ran 
 - present figure of pred vs absolute
 -- can explain that this helps show where the differences are
 - show code of loop for all models
 - discuss possibility of grid search for optimizing each model parameter
 - 
 
