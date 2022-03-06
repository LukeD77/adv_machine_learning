# Assignment 4 - Repeated boosting algorithm and LightGBM

### LightGBM
LightGBM differs from XGBoost as it uses Gradient-based One Side Sampeling (GOSS), which only utilizes a small portion of the data to calculate the solution gradient. The data excluded are those that create a smaller gradient compared to the rest of the data.  The second difference from XGBoost is feature selection via a process called Exclusive Feature Bundling (EFB). The features chosen are those that are not found with zero values simultaneously of other values. With these streamlining features, LightGBM is reported to increase prediction speed over 20 fold that of XGBoost. 

### Code for LightGBM
```python
gbm = lgb.LGBMRegressor(num_leaves = 200, n_estimators=50, learning_rate = 0.1, max_depth = 50)
gbm_fit = gbm.fit(xtrain, ytrain, eval_set=(xtest,ytest), verbose=0)
```
In this assignment I tested Lowess regression or XGBoost as the primary regression model and used a decision tree classifier to boost the primary model. I ran 1, 2, or 3 boostings for each model type. I also included LightGBM in the loop model run. The data used was the concrete stregth data with the following input variables:
 - Cement (component 1)(kg in a m^3 mixture)
 - Blast Furnace Slag (component 2)(kg in a m^3 mixture)
 - Fly Ash (component 3)(kg in a m^3 mixture)

And an output variable of Concrete compressive strength(MPa, megapascals). The table below (table 1) provides MSE for the model outputs for each model output along with number of boostings.

**Table 1**


| Model | Boost Number    | MSE
| :---:  | :---: | :---: |
| Lowess Reg | 1 | 149.00
| Lowess Reg | 2 | 149.40
| Lowess Reg | 3 | 149.85
| XGB | 1 | 158.30
| XGB | 2 | 158.42
| XGB | 3 | 158.09
|LightGBM | 0 | 154.17


The Lowess Regression overall acheived the best accuracy with boostings 1,2, and 3 in order of best to worst accuracy. Interstingly, XGB had the best accuracy at 3 boostings, then 1, then 2. LightGBM was in the middle in terms of accuracy. I initially attempted to run all 8 input variables through the models, but Lowess Regression alone took almost an hour, so I reduced the amount of features to 3. However, LightGBM's value is by reducing the number of features to increase speed. Therefore, I attempted to run the model with 3 features and 8 features, and then compare it to running XGboost with zero boosting to compare the time taken as well as model accuracy. The results are found below in table 2.

**Table 2**

| Model |Train data size |  MSE | time (seconds)
| :---: | :---: | :---: | :---: |
| LightGBM | Small | 163.56 | 0.0279
| LightGBM | Large | 34.43 | 0.0319 
| XGBoost | Small | 180.95 | 0.2264
| XGBoost | Large | 31.07 | 0.2832

Numerous conclusions can be drawn from this test. Firstly, the inclusing of the larger dataset does not severely hinder the runtime of either LightGBM or XGBoost, with only a meager difference between the small and large dataset runtimes for each model. Secondly, LightGB performs almost 10 times faster than XGBoost. However, the speed may come at a cost of accuracy. While LightGBM was more accurate than XGBoost using the 3 feature dataset, XGBoost was more accurate on the 8 feature dataset. My hypothesis is that the exclusion of certain features in LightGBM removed informative data that was included in the predictions of XGBoost. In this case, XGBoost was more useful in the larger dataset, contrary to the engineered intent of LightGMB.

The code below is the large model loop used to generate table 1.
```python
num_repeats = 2

mse_all_list1 = []
mse_avg_list1 = []
mse_all_list2 = []
mse_avg_list2 = []
mse_all_list3 = []
mse_avg_list3 = []

mse_all_list1_xgb = []
mse_avg_list1_xgb = []
mse_all_list2_xgb = []
mse_avg_list2_xgb = []
mse_all_list3_xgb = []
mse_avg_list3_xgb = []

mse_all_list_gbm = []
mse_avg_list_gbm = []

for i in range(num_repeats):
    kf = KFold(n_splits= 10, shuffle = True, random_state = i)
    for idxtrain, idxtest in kf.split(X):
        xtrain = X[idxtrain]
        ytrain = y[idxtrain]
        ytest = y[idxtest]
        xtest = X[idxtest]
        xtrain = scale.fit_transform(xtrain)
        xtest = scale.transform(xtest)
        
        # Running boosted lowess with 1, 2, and 3 boostings
        yhat_1 = boosted_lowess_reg(xtrain, ytrain, xtest, tricubic, tau=0.5, intercept= True, 
                   model_boosting = model_boosting, numboost = 1)
        mse_all_list1.append(mse(ytest, yhat_1))
        
        yhat_2 = boosted_lowess_reg(xtrain, ytrain, xtest, tricubic, tau=0.5, intercept= True, 
                   model_boosting = model_boosting, numboost = 2)
        mse_all_list2.append(mse(ytest, yhat_2))

        yhat_3 = boosted_lowess_reg(xtrain, ytrain, xtest, tricubic, tau=0.5, intercept= True, 
                   model_boosting = model_boosting, numboost = 3)
        mse_all_list3.append(mse(ytest, yhat_3))
    
        # Running xgb with 1, 2 and 3 boostings 
        output1 = booster_xgb(xtrain,ytrain,xtest,model_boosting,1)
        mse_all_list1_xgb.append(mse(ytest, output1))
        
        output2 = booster_xgb(xtrain,ytrain,xtest,model_boosting,1)
        mse_all_list2_xgb.append(mse(ytest, output2))
        
        output3 = booster_xgb(xtrain,ytrain,xtest,model_boosting,1)
        mse_all_list3_xgb.append(mse(ytest, output3))
    
    
        # Running lightgbm 
        gbm = lgb.LGBMRegressor(num_leaves = 200, n_estimators=50, learning_rate = 0.1, max_depth = 50)
        gbm_fit = gbm.fit(xtrain, ytrain, eval_set=(xtest,ytest), verbose=0, )
        yhat_gbm = gbm_fit.predict(xtest)
        mse_all_list_gbm.append(mse(ytest, yhat_gbm))
        
    mse_avg_list1.append(np.mean(mse_all_list1))
    mse_all_list1 = []
    mse_avg_list2.append(np.mean(mse_all_list2))
    mse_all_list2 = []
    mse_avg_list3.append(np.mean(mse_all_list3))
    mse_all_list3 = []
    
    mse_avg_list1_xgb.append(np.mean(mse_all_list1_xgb))
    mse_all_list1_xgb = []
    mse_avg_list2_xgb.append(np.mean(mse_all_list2_xgb))
    mse_all_list2_xgb = []
    mse_avg_list3_xgb.append(np.mean(mse_all_list3_xgb))
    mse_all_list3_xgb = []
    
    mse_avg_list_gbm.append(np.mean(mse_all_list_gbm))
    mse_all_list_gbm = []
```
