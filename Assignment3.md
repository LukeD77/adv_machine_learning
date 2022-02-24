# Assignment 3 - Multivariate Regression Analysis and Gradient Boosting

### Extreme gradient boost

Extreme Gradient Boost (XGBoost) is a method of machine learning that uses decision trees which create splits in the test data and permforms calculations, including the residuals, for making new decision tree leaves. The calculation used is called the gain function, the equation of which is below.

![gain function](https://user-images.githubusercontent.com/67921793/155613003-5e788a89-61ae-4e02-a7ad-373cd380c50f.png)

- **G<sub>L</sub>** is the sum of residuals on the left split
- **G<sub>R</sub>** is the sum of residuals on the right split
- **H<sub>R</sub>** is the number of residuals on the right split
- **H<sub>L</sub>** is the number of residuals on the left split
- **λ** is the regularization parameter which makes the individual observations less influencial to model predictions
- **γ** is the required loss to split a node into two more leaves on the model

After all splits are made and gain functions are evaluated, the split yielding the largest gain function is used to make a new decision tree leaf. When all splits yield negative gain values, then the tree reaches the end. This tree then predicts on the train data and new residuals are created to train another tree. This process is repeated until the number of estimators is reached. Then the sum of each decision tree times the learning rate becomes the final prediction of the test data.

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

The models used the Boston Housing Price data with crime rate, number of rooms, and distance to predict the housing price. All models underwent manual optimization of hyperparameters and kernels when required. No automated grid searching was conducted. After running the models through 5 split kfold 50 times, the following MSE and MAE values were produced. 
 
| Model | MSE    | MAE
| :---:   | :---: | :---: |
| Lowess Reg | 28.65     | 3.29
| Boosted Lowess Reg | 26.20 | 3.31
| Random Forest | 26.19 | 3.20
| XGB | 22.08 | 3.11
 
 
Based upon the MAE and MSE values, XGB performed the best, there was a tie between Random Forest and Boosted Lowess Regression for 2nd, and Lowess Regression was last. While these values are helpful to interpret model accuracy, they do not lend any indication for what areas of the data the model struggled with. For this, I plotted the predicted vs actual house prices for the last of the 50 iterations for each model.

![pred vs actual](https://user-images.githubusercontent.com/67921793/155621826-80c5458b-e13c-423f-92a2-639627b8ad24.png)

These figures, while rather chaotic, show that the models struggle to predict values towards the middle of the range of housing prices, with vast differences between close values. The model also does not effectively predict the price of expensive houses as the houses past the predicted cost of 35 are all lower than the actual. However, these figures are quite messy and the difference between the two values is not easily discernable. Therefore, I plotted the difference between predicted and actual y values across the 500 samples of the last iterations.

![assign3_2](https://user-images.githubusercontent.com/67921793/155622651-b041cae6-a9d3-40c9-8b17-51a6def98d9a.png)

These graphs display a clearer depiction of the predictive power, or lack there of, for the models in certain areas. These values are sorted, so the leftmost values are lower prices and the rightmost values are the highest prices. The models all struggle to predict small house prices but then slwly decrease in the highest and overall differences between predicted and actual as you move into the median price houses. The models then at the end start to get more errnoeous predicting high priced houses, as was evident in the previous figure. Ideally, these graphs would not be used since the residual would be close to zero, but these models struggle to accurrately predict such clustered data. Different approaches will be required to model these data more accurately.

**Below is the code for running the models 50 times with the kfold.**

 ```python
 mse_lwr = []
mse_boostlwr = []
mse_rf = []
mse_boostrf = []

mae_lwr = []
mae_boostlwr = []
mae_rf = []
mae_boostrf = []

yhat_lwr_all = []
yhat_boostlwr_all = []
yhat_rf_all = []
yhat_xgb_all = []

num_repeats = 50
n_splits = 5

for i in range(num_repeats):
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=i)
    for idxtrain, idxtest in kf.split(X):
        xtrain = X[idxtrain]
        ytrain = y[idxtrain]
        xtest = X[idxtest]
        ytest = y[idxtest]
        xtrain = scale.fit_transform(xtrain)
        xtest = scale.transform(xtest)
        
        # Running Lowess Regression
        yhat_lwr = lowess_reg(xtrain, ytrain, xtest, quartic, tau=0.3, intercept=True)
        
        # Running Boosted Lowess Regression
        yhat_boostlwr = boosted_lowess_reg(xtrain,ytrain,xtest,quartic,tau=1,intercept=True)
        
        # Running Random Forest
        model_rf = RandomForestRegressor(n_estimators = 300, max_depth = 7)
        model_rf.fit(xtrain, ytrain)
        yhat_rf = model_rf.predict(xtest)
        
        # Running boosted Random Forest
        model_boostrf = xgb.XGBRegressor(objective='reg:squarederror',
                                         n_estimators=300,reg_lambda=20,alpha=1,gamma=10,max_depth=7)
        model_boostrf.fit(xtrain,ytrain)
        yhat_boostrf = model_boostrf.predict(xtest)
        
        # Getting mse and mae
        mse_lwr.append(MSE(ytest,yhat_lwr))
        mse_boostlwr.append(MSE(ytest,yhat_boostlwr))
        mse_rf.append(MSE(ytest,yhat_rf))
        mse_boostrf.append(MSE(ytest,yhat_boostrf))
        
        mae_lwr.append(MAE(ytest,yhat_lwr))
        mae_boostlwr.append(MAE(ytest,yhat_boostlwr))
        mae_rf.append(MAE(ytest,yhat_rf))
        mae_boostrf.append(MAE(ytest,yhat_boostrf))
        
        if i == num_repeats-1:
            #print('adding')
            for a in yhat_lwr:
                yhat_lwr_all.append(a)
            for a in yhat_boostlwr:
                yhat_boostlwr_all.append(a)
            for a in yhat_rf:
                yhat_rf_all.append(a)
            for a in yhat_boostrf:
                yhat_xgb_all.append(a)

print('The Cross-Validated MSE for LWR is: ' + str(np.mean(mse_lwr)))
print('The Cross-Validated MSE for BOOST LWR is: ' + str(np.mean(mse_boostlwr)))
print('The Cross-Validated MSE for RF is: ' + str(np.mean(mse_rf)))
print('The Cross-Validated MSE for XGB is: ' + str(np.mean(mse_boostrf)))

print(' ')
        
print('The Cross-Validated MAE for LWR is: ' + str(np.mean(mae_lwr)))
print('The Cross-Validated MAE for BOOST LWR is: ' + str(np.mean(mae_boostlwr)))
print('The Cross-Validated MAE for RF is: ' + str(np.mean(mae_rf)))
print('The Cross-Validated MAE for XGB is: ' + str(np.mean(mae_boostrf)))
 ```
