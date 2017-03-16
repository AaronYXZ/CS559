import CS559_HW2_setup as cs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns


m = cs.Neural_Network()
X,y = m.makeData(0,2,600,3)
Xt,yt = m.makeData(0,2,600,3)

## plot against different learning_rate
#pp = [0.1, 0.05,0.02, 0.01,0.005,0.002, 0.001,0.0005,0.0002, 0.0001]
pp = np.linspace(0,0.001,21)
result1 = pd.DataFrame(np.zeros((len(pp), 3)))
result1.columns = ['LR', 'Train_error','Test_error']
result1['LR'] = pp
result2 = pd.DataFrame(np.zeros((len(pp), 3)))
result2.columns = ['LR', 'Train_logloss','Test_logloss']
result2['LR'] = pp
for i, p in enumerate(pp):
    m = cs.Neural_Network(50,p,10000,0.1)
    V,W = m.fit(X,y)
    train_error = m.errors(V,W,X, y)
    test_error = m.errors(V,W,Xt, yt)
    result1.ix[i, 'Train_error'] = train_error
    result1.ix[i, 'Test_error'] = test_error
    train_logloss = m.cross_entropy(V,W,X,y)
    test_logloss = m.cross_entropy(V,W,Xt,yt)
    result2.ix[i, 'Train_logloss'] = train_logloss
    result2.ix[i, 'Test_logloss'] = test_logloss

## plot against different hiddent_units
# hu = [2,5,10,20,30,50,100]
hu = np.arange(0, 100, 5)
result3 = pd.DataFrame(np.zeros((len(hu), 3)))
result3.columns = ['HU', 'Train_error','Test_error']
result3['HU'] = hu
result4 = pd.DataFrame(np.zeros((len(hu), 3)))
result4.columns = ['HU', 'Train_logloss','Test_logloss']
result4['HU'] = hu
for i, u in enumerate(hu):
    m = cs.Neural_Network(u,0.001,10000,0.1)
    V,W = m.fit(X,y)
    train_error = m.errors(V,W,X, y)
    test_error = m.errors(V,W,Xt, yt)
    result3.ix[i, 'Train_error'] = train_error
    result3.ix[i, 'Test_error'] = test_error
    train_logloss = m.cross_entropy(V,W,X,y)
    test_logloss = m.cross_entropy(V,W,Xt,yt)
    result4.ix[i, 'Train_logloss'] = train_logloss
    result4.ix[i, 'Test_logloss'] = test_logloss

figure = plt.figure(figsize = (24,8))
ax1 = plt.subplot(2,2,1)
# ax1.plot(np.log10(result1['LR']), result1['Train_error'], linestyle='--')
# ax1.plot(np.log10(result1['LR']), result1['Test_error'])
ax1.plot(result1['LR'], result1['Train_error'], linestyle='--')
ax1.plot(result1['LR'], result1['Test_error'])
ax1.set_xlabel('Learning_Rate')
ax1.set_ylabel('Misclassification_Error')
ax1.legend(loc = 2)

ax2 = plt.subplot(2,2,2)
# ax2.plot(np.log10(result2['LR']), result2['Train_logloss'])
# ax2.plot(np.log10(result2['LR']), result2['Test_logloss'])
ax2.plot(result2['LR'], result2['Train_logloss'], linestyle='--')
ax2.plot(result2['LR'], result2['Test_logloss'])
ax2.set_xlabel('Learning_Rate')
ax2.set_ylabel('Cross_Entropy')
ax2.legend(loc = 2)

ax3 = plt.subplot(2,2,3)
ax3.plot(result3['HU'], result3['Train_error'], linestyle='--')
ax3.plot(result3['HU'], result3['Test_error'])
ax3.set_xlabel('Hidden_Units')
ax3.set_ylabel('Misclassification_Error')
ax3.legend(loc = 2)

ax4 = plt.subplot(2,2,4)
ax4.plot(result4['HU'], result4['Train_logloss'], linestyle='--')
ax4.plot(result4['HU'], result4['Test_logloss'])
ax4.set_xlabel('Hidden_Units')
ax4.set_ylabel('Cross_Entropy')
ax4.legend(loc = 2)

plt.show()


if __name__ == '__main__':
    pass