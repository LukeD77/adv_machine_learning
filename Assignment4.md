# Assignment 4 - Repeated boosting algorithm and LightGBM

### LightGBM
LightGBM differs from XGBoost as it uses Gradient-based One Side Sampeling (GOSS), which only utilizes a small portion of the data to calculate the solution gradient. The data excluded are those that create a smaller gradient compared to the rest of the data.  The second difference from XGBoost is feature selection via a process called Exclusive Feature Bundling (EFB). The features chosen are those that are not found with zero values simultaneously of other values. With these to streamlining features, LightGBM is reported to increase prediction speed over 20 fold. 

### Code for LightGBM
```python
gbm = lgb.LGBMRegressor(num_leaves = 200, n_estimators=50, learning_rate = 0.1, max_depth = 50)
gbm_fit = gbm.fit(xtrain, ytrain, eval_set=(xtest,ytest), verbose=0, )
```
In this assignment I tested Lowess regression or XGBoost as the primary regression model and used a decision tree classifier to boost the primary model. I ran 1, 2, or 3 boostings for each model type. I also included LightGBM in the loop model run. The table below provides the output for their MSE.

