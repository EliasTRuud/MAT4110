import numpy as np

A = np.array([[1, 2, 3], [0, 3, 4], [0, 0, 5]])
b = [1,1,1]

def back_sub(A, b):
    n = np.shape(A)[0]
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        temp_sum = 0
        for j in range(i+1, n):
            temp_sum += A[i, j] * x[j]
        x[i] = (b[i] - temp_sum) / A[i, i]
    return x


x = back_sub(A, b)
x_test = np.linalg.solve(A,b)

print(x)
print(x_test)


# Can get large error, can look at matrix with np.linalg.cond(A). Condition number big is bad
# If large can try LU decomposition