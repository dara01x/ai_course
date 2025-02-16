from sklearn import datasets

# Load the Boston Housing dataset (deprecated, use fetch_california_housing instead)
# boston = datasets.load_boston()
# X_boston = boston.data
# y_boston = boston.target
# print('\nBoston Housing dataset:')
# print('X-size = ', X_boston.shape)
# print('y-size = ', y_boston.shape)
# print(X_boston[0:5, :])

# Load the Breast Cancer dataset
breast_cancer = datasets.load_breast_cancer()
X_breast_cancer = breast_cancer.data
y_breast_cancer = breast_cancer.target
print('\nBreast Cancer dataset:')
print('X-size = ', X_breast_cancer.shape)
print('y-size = ', y_breast_cancer.shape)
print(X_breast_cancer[0:5, :])

# Load the Diabetes dataset
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
print('\nDiabetes dataset:')
print('X-size = ', X_diabetes.shape)
print('y-size = ', y_diabetes.shape)
print(X_diabetes[0:5, :])

# Load the Digits dataset
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print('\nDigits dataset:')
print('X-size = ', X_digits.shape)
print('y-size = ', y_digits.shape)
print(X_digits[0:5, :])

# Load the Iris dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
print('\nIris dataset:')
print('X-size = ', X_iris.shape)
print('y-size = ', y_iris.shape)
print(X_iris[0:5, :])

# Load the California Housing dataset
california_housing = datasets.fetch_california_housing()
X_california = california_housing.data
y_california = california_housing.target
print('\nCalifornia Housing dataset:')
print('X-size = ', X_california.shape)
print('y-size = ', y_california.shape)
print(X_california[0:5, :])
