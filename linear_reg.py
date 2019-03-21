import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv(r'D:\Users\Michelle\AppData\Local\Programs\Python\Python36\data\train.csv')
df_test = pd.read_csv(r'D:\Users\Michelle\AppData\Local\Programs\Python\Python36\data\test.csv')

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def variance(numbers, mean):
    return sum([abs(x-mean)**2 for x in numbers])

def covariance(x_train,x_mean, y_train, y_mean):
    ln = len(x_train)
    cov = 0.0
    for i in range(ln):
        cov += ((x_train[i] - x_mean) * (y_train[i] - y_mean))
    return cov

def coefficients():
    m = covariance(x_test,x_mean, y_test, y_mean) / variance(x_test, x_mean)
    b = y_mean - (m*x_mean)
    return [m,b]

def simple_linear_regression():
    prediction = []
    m, c = coefficients()
    for test in x_test:
        y_pred = m*test[0] + c
        prediction.append(y_pred)
    return prediction

print(df_train.head())
print(df_test.head())
print(df_train.shape)
print(df_test.shape)

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

#print(x_train.head())
#print(y_train.head())
#print(x_test.head())
#print(y_test.head())

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(np.where(np.isnan(df_train['x'])))
print(np.where(np.isnan(df_train['y'])))
df_train = df_train.drop(213) #drop nan value

x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#print(x_test)
#print(y_test)

x_mean, y_mean = mean(x_test), mean(y_test)
x_var = variance(x_test,x_mean)
y_var = variance(y_test, y_mean)

predict = simple_linear_regression()
#print(predict)

plt.plot(x_test, predict, c='red', label='Regression Line')
plt.scatter(x_train, y_train, label='data', c='blue')

plt.xlabel('Independent variable (x)')
plt.ylabel('Dependent variable (y)')
plt.legend()
plt.show()
