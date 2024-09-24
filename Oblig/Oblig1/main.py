import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Vandermonde matrix
def make_vander(x, n):
    '''
    x: vector input, 1D array
    n: number of columns in the output, power of x minus 1. (m = 3 -> max power = x**2)
    '''
    return np.vander(x, n, increasing=True)

# Back substitution
def back_substitution(A, b):
    '''
    A: Square upper triangular matrix (n x n)
    b: 1D vector (n)

    Solves the system by starting from the last row of the matrix (the simplest equation)
    and working upwards to solve for each unknown
    '''
    n = np.shape(A)[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        temp_sum = 0
        for j in range(i + 1, n):
            temp_sum += A[i, j] * x[j]
        x[i] = (b[i] - temp_sum) / A[i, i]
    return x

def QR_solve(A, b):
    ''' 
    A: Matrix (n x m) 
    b: vector (n), right side of the equation system

    Decomposition with QR from numpy, then use backsub to find solutions x
    The solution is found by solving the system Rx = Q_T@b using back substitution.
    Which is efficient due to R being upper triangular.
    '''
    Q, R = np.linalg.qr(A)  # returns orthogonal matrix Q and upper triangular matrix R
    b_ = np.dot(Q.T, b) # Rx = b_, just used b_ to so it seems similar to Ax=b
    x = back_substitution(R, b_)
    return x

# Generate noisy data sets
n = 30
start = -2
stop = 2
eps = 1
np.random.seed(1) # set seed for number gen
x = np.linspace(start, stop, n)
r = np.random.rand(n) * eps # random noise which is added to both signals

# First dataset
y1 = x*(np.cos(r + 0.5*x**3) + np.sin(0.5*x**3))
y1_noiseless = x*(np.cos(0.5*x**3) + np.sin(0.5*x**3))

# Second dataset
y2 = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r
y2_noiseless = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10

#QR solving
# Degrees of poly -1 (dimension of vander matrix)
m1 = 3  # 2nd degree poly
m2 = 8  # 7th degree poly

# Vandermonde with dataset
A_1 = make_vander(x, m1)
A_2 = make_vander(x, m2) 

# Fit the first dataset (y1)
x1_QR_3 = QR_solve(A_1, y1)
y1_pred_QR_3 = np.dot(A_1, x1_QR_3)

x1_QR_8 = QR_solve(A_2, y1)
y1_pred_QR_8 = np.dot(A_2, x1_QR_8)

# Fit the second dataset (y2)
x2_QR_3 = QR_solve(A_1, y2)
y2_pred_QR_3 = np.dot(A_1, x2_QR_3)

x2_QR_8 = QR_solve(A_2, y2)
y2_pred_QR_8 = np.dot(A_2, x2_QR_8)


"""
#Compare with numpy solution
x1_numpySol, _, _, _ = np.linalg.lstsq(A_1, y1)
y1_sol = np.dot(A_1, x1_numpySol)
"""

#Calculate MSE
#MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 


#Method 2: Solve normal equations using Cholesky factorization: A = L D L_T
#          where L is lower triangular matrix and D is a diagonal matrix


# We implement the Cholesky factorization and forward substitution functions from the group sessions
# The solution of the normal equations is likely to be unstable. Therefore this
# method is not recommended in general. For small problems it is usually safe to use
# https://www.math.iit.edu/~fass/477577_Chapter_5.pdf

def cholesky(A):
    n = np.shape(A)[0]
    if n != np.shape(A)[1]:
        raise Exception("Matrix is not square")
    
    if not np.all(np.abs(A - A.T) < 1e-10):
        raise Exception("Matrix is not symmetric")

    L, D = np.zeros((n, n)), np.zeros((n, n))

    A_k = A
    for k in range(0, n):
        if np.abs(A_k[k, k]) <= 1e-10:
            raise Exception("Matrix is singular")
        
        L[:, k] = A_k[:, k] / A_k[k, k]
        D[k, k] = A_k[k, k]
        A_k = A_k - D[k, k] * (L[:, k : k + 1] @ L[:, k : k + 1].T)

    return L, D

def chol_solve(A, b):
    '''
    A: matrix (n, m)
    b: vector (n), which is the targets you want to estimate
    returns: solution for x which minimizes the least squares solution

    Solve the linear equation system where Ax = b then multiply A_T left side.
    B = A_T@A -> B*x = A_T*b = y
    Apply cholesky to B which is symmetric: B = L*D*L_T
    Seperate into L * (D*L_T) * x = L*z = y and do forward sub to find z = D L_t x = D R x
    Then we do back sub to find x from equation for z.
    '''
    B = A.T @ A #make a symmetric matrix
    y = A.T @ b # R.T * z = y

    cond = np.linalg.cond(B)  
    if(cond > 1e4):
        print(f'Warning: Ill conditioed matrix B might give bad results, condition number = {cond:.0f}')

    L, D = cholesky(B)
    #L1 = np.linalg.cholesky(B) #tested if correctly found L
  
    R = L.T #rename to R, as L.T

    # Solve for z in, R.T * z = y. (lower tri -> forward sub), where z =  D R x
    z = forward_substitution(R.T, y)

    #Then solve for x in, D R * x = z
    x = back_substitution(D@R, z)
    
    return x


def forward_substitution(A, b):
    '''
    L : A lower triangular matrix of shape (n, n)
    b : 1D array (n), right hand side of system Lx=b
    returns x: (n), solution vector

    Similar to backward_substitution, however loop starts at 0 as we iterate downwards
    in the matrix and solve the lower triangular system, A x = b.
    '''
    n = np.shape(A)[0]
    x = np.zeros(n)

    for i in range(n):
        temp_sum = 0
        for j in range(i):
            temp_sum += A[i, j] * x[j]
        x[i] = (b[i] - temp_sum) / A[i, i]

    return x

def mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


# Fit the first dataset (y1) using Cholesky fac
x1_chol_3 = chol_solve(A_1, y1)
y1_pred_chol_3 = np.dot(A_1, x1_chol_3)

x1_chol_8 = chol_solve(A_2, y1)
y1_pred_chol_8 = np.dot(A_2, x1_chol_8)

#  Second dataset (y2) 
x2_chol_3 = chol_solve(A_1, y2)
y2_pred_chol_3 = np.dot(A_1, x2_chol_3)

x2_chol_8 = chol_solve(A_2, y2)
y2_pred_chol_8 = np.dot(A_2, x2_chol_8)



# MSE for different methods
#Set 1
mse_QR_3 = mse(y1, y1_pred_QR_3)
mse_chol_3 = mse(y1, y1_pred_chol_3)
mse_QR_8 = mse(y1, y1_pred_QR_8)
mse_chol_8 = mse(y1, y1_pred_chol_8)

mse_QR_3 = mse(y2, y2_pred_QR_3)
mse_chol_3 = mse(y2, y2_pred_chol_3)
mse_QR_8 = mse(y2, y2_pred_QR_8)
mse_chol_8 = mse(y2, y2_pred_chol_8)

#MSE metric but didnt notice any difference between them
print(f"Dataset 1, m=3, MSE: QR = {mse_QR_3:.2e}, Chol = {mse_chol_3:.2e}")
print(f"Dataset 1, m=8, MSE: QR = {mse_QR_8:.2e}, Chol = {mse_chol_8:.2e}")

print(f"Dataset 2, m=3, MSE: QR = {mse_QR_3:.2e}, Chol = {mse_chol_3:.2e}")
print(f"Dataset 2, m=8, MSE: QR = {mse_QR_8:.2e}, Chol = {mse_chol_8:.2e}")


# Plotted for both methods and datasets with m=3 and m=8
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12, 12))
ax0, ax1, ax2, ax3 = axs.flatten()
sns.set_theme()
fig.suptitle('Least Squares QR vs Cholesky', fontsize=18)

# Dataset 1
ax0.plot(x, y1, 'o', label='Data (y1)')
ax0.plot(x, y1_pred_QR_3, '-', label='QR(m=3)')
ax0.plot(x, y1_pred_chol_3, '--', label='Cholesky (m=3)')
ax0.set_title('Dataset 1, m=3')
ax0.legend()

ax2.plot(x, y1, 'o', label='Data (y1)')
ax2.plot(x, y1_pred_QR_8, '-', label='QR (m=8)')
ax2.plot(x, y1_pred_chol_8, '--', label='Cholesky(m=8)')
ax2.set_title('Dataset 1, m=8')
ax2.legend()

# Dataset 2
ax1.plot(x, y2, 'o', label='Data (y2)')
ax1.plot(x, y2_pred_QR_3, '-', label='QR (m=3)')
ax1.plot(x, y2_pred_chol_3, '--', label='Cholesky (m=3)')
ax1.set_title('Dataset 2, m=3')
ax1.legend()

ax3.plot(x, y2, 'o', label='Data (y2)')
ax3.plot(x, y2_pred_QR_8, '-', label='QR (m=8)')
ax3.plot(x, y2_pred_chol_8, '--', label='Cholesky (m=8)')
ax3.set_title('Dataset 2, m=8')
ax3.legend()

plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig("QR_vs_Cholesky.pdf")
# plt.show()
plt.clf()


