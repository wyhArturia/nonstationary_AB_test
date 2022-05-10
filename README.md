# KDD_2022_code

This code is for our KDD paper ``non-stationary AB testing''. Although there are several files, only the parameters are different and the main parts are the same. 

The file ``ABtest_N=8000_T=100000.py'' is annotated, while others are not, so we strongly recommend to use this code for numerical experiments.

In the code, T represents the number of total iterations. To get an accurate estimator of MSE, we set T=100000 in the code, which can take several hours when n is large. 

We suggest to use T<=100 if you just want rough results.

Because of the data privacy policy of the company, we can't release the code for real data but only simulations on calibrated data.

If you have any question, please contact us after the double-blinded review.
