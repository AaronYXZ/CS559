How to run the code:

Put CS559_HW2_run.py and CS559_HW2_setup.py in the same directory. In Mac CL, type “python CS559_HW2_run.py”, it will show graphs with the cross-entropy error/misclassification error with regard to learning rate η and number of hidden units, 
for both training and test sets(This should take about 5 minutes). 


How to interpret the results:

As shown by the graphs (1 & 2), both the training error and test error decreases as the learning rate decreases. Since in my code the smallest learning rate is set to 0.0005 and the lowest error is achieved at this point, I also did another test by setting the learning rate η to be even smaller, ranging from 0.0001 to 0.001. Again the smallest test error is obtained at the smallest η. I don't know if there exist an optimal η, or η should be set as small as possible to achieve the best test performance. 

In graph 3 & 4, the best test performance is obtained where hidden units = 40.  This shows that having more hidden units doesn't necessarily guarantee a better performance. 
