# Assignment 5

Creating synthetic datasets, creating SCAD and SQRTLasso sklearn structured functions, and using GridSearch optimization.

| Model | Best Estimator | Mean number of true coeff (27 max) | Mean L2 Distance | Mean RMSE
| :---:  | :---: | :---: | :---:  | :---: |
| SQRtLasso | alpha = 0.16 | 27 | 1.48 | 3.40 |
| SCAD | a = 2, lam = 0.05  |27 |3.17 |5.51 |
| Lasso | alpha = 0.37 | 18.85 | 2.72 | 3.44 |
| Ridge | alpha =  0.01 | 27.0 | 3.05 | 5.19 |
| ElasticNet | alpha = 0.53, l1_ratio = 0.79 | 23.34 |3.02 |3.87 |


The results show a few interesting patterns. Firstly, the models with Lasso and ElasticNet did not recoup all 27 non-zero coefficients, whereas Ridge, SQRTLasso, and SCAD did. It is interesting that SQRTLasso did not zero out the 27 coefficients while minimizing to E^-10 values for the other coefficients. Ridge and SCAD, I hypothesize, did not zero out any coefficients and that is why they retained all 27. Even though the Lasso and ElasticNet did not get all 27 non-zero coefficients, they still had a higher accuracy than Ridge and SCAD, as illustrated by the lower mean RMSE scores. SQRTLasso had the highest accuracy in terms of identifying the number of true coefficients (when comparing those that did any sort of variable selection), the smallest L2 distance by a large margin, and the smallest RMSE. Notably, SCAD and Ridge has similiar RMSE values and as did SQRTLasso, Lasso, and Elastic Net. 
