import numpy as np
from scipy.optimize import minimize

# Ground truth log-values: y(0,0)=0, y(0,1)=1, y(1,0)=2, y(1,1)=4
# Approximation: q(x1,x2) = theta1*x1 + theta2*x2 + b
# Unnormalized KL: sum[ exp(y_i)*(y_i - q_i) - exp(y_i) + exp(q_i) ]

def objective(params):
    t1, t2, b = params
    # q values for each state
    q = [b, t2 + b, t1 + b, t1 + t2 + b]
    y = [0, 1, 2, 4]
    return sum(np.exp(yi) * (yi - qi) - np.exp(yi) + np.exp(qi) for yi, qi in zip(y, q))

result = minimize(objective, x0=[0.0, 0.0, 0.0], method='Nelder-Mead')
print(f"theta1 = {result.x[0]:.6f}")
print(f"theta2 = {result.x[1]:.6f}")
print(f"b      = {result.x[2]:.6f}")
print(f"min value = {result.fun:.6f}")
print(f"success: {result.success}")
print()
# Verify: show approximation vs ground truth
t1, t2, b = result.x
print("State  | y (true) | q (approx)")
for (s1, s2), yi in zip([(0,0),(0,1),(1,0),(1,1)], [0,1,2,4]):
    qi = t1*s1 + t2*s2 + b
    print(f"({s1},{s2})  |    {yi}     |  {qi:.4f}")
