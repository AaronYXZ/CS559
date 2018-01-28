

## How to run the code:

Put CS559_HW1_run.py and CS559_HW1_setup.py in the same directory, along with the cleaned data hw1_train.csv and hw1_test.csv.
(Data cleaning process is completed using HW1_playground.ipynb).
In Mac CL, type “python CS559_HW1_run.py”, this should return the test error: 80.7%.

## How to get weight vecotr:

In python, import CS559_HW1_setup.py, create a model object using this code: "model = ch.Logistic_Regression(0.01, 0, 200)", 
then get weight vector using this code: "model.theta(X_train, y_train)".

## How to interpret the results:

The test error of logistic regression is 80.7%, but a closer look at the predict values of y reveals that the model has predict
all the test data to be class of 0. This is under-fitting, probably because the data itself is not iid: identically, 
independently distributed. I also implemented a plot method which can plot the changes of weight vector-theta over iteration. It
shows that most theta converge to a negative value, thus giving a large sigmoid value that leads to close to zero predicted 
probability. Sklearn also gives a similar result, which kinda confirms my belief that our traing data violiates the assumptions 
of the logistic regression model. 
