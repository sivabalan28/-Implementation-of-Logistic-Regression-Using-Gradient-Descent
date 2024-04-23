# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SIVABALAN S
RegisterNumber: 212222240100
```
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
### Array Value of x
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/d9ab87e8-58dc-436a-adba-0390436d26a8)

### Array Value of y
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/c3dfd3a6-7677-458c-9937-16409940f1f9)

### Exam 1 - score graph
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/cbe518eb-4fa5-4f6b-8fa6-1410339e2759)

### Sigmoid function graph
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/0d6714d9-a6fb-4605-a446-90c9ff6bb1fa)

### X_train_grad value
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/ece6116e-1299-46dc-8030-475fc261b6cf)

### Y_train_grad value
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/92fd22bf-778d-47d6-b35d-50ac2f5d95a8)

### Print res.x
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/7e482e86-e94f-488d-b25d-ae6c5969c3b1)

### Decision boundary - graph for exam score
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/cd203767-73de-4273-93a7-5036115bca0b)

### Proability value
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/4bbf5019-b6d1-420d-a1b8-91cce3eb222e)

### Prediction value of mean
![image](https://github.com/KameshLeVI/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120780633/2d042d26-483e-473a-9415-9ca7f917420e)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

