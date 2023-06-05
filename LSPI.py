'''
LSPI
'''
import numpy as np
from scipy.optimize import minimize

def LSPI(basis_functions, gamma, epsilon, w, env, method = "discrete", n_trial_samples = 1000, n_timestep_samples=20):
    '''
    通过LSPI算法计算policy的参数矩阵w

    Inputs:
    sample: 由list组成的tuple (s,a,r,s')
    basis_functions: basis函数的list
    gamma: 折扣因子，float类型
    epsilon: 收敛阈值，float类型
    w: 初始化策略参数矩阵
    
    Outputs:
    w: 收敛后的策略参数矩阵
    '''
    w0 = []
    
    samples = _generate_samples(env, n_trial_samples, n_timestep_samples)

    while True:
        w_prev = w

        w = _LSTDQ_OPT(samples, basis_functions, gamma, w, env, method=method)
        
        if _converged(w, w_prev, epsilon):
            break
        else:
            w_prev = w
        w0.append(w[0])
        print (w[0])
    return w, w0


def get_policy_action(s,w, basis_functions, env, method="discrete"):
    '''

    根据提供的方法（离散、连续或连续离散）计算当前参数化策略的最佳action
    
    Inputs:
    s: state
    w: policy matrix
    basis_functions: basis function list
    env: gym
    method: string
    '''
    if method == "discrete":
        return _get_policy_action_discrete(s,w,basis_functions,env)
    if method == "continuous":
        return _get_policy_actions_continuous(s, w, basis_functions, env)
    if method == "continuous-discretized":
        return _get_policy_actions_continuous_discretized(s, w, basis_functions, env, n_discretizations=10)


### 私有方法
def _converged(w, w_prev, epsilon):
    return np.linalg.norm(w - w_prev) < epsilon

def _LSTDQ(samples, basis_functions, gamma, w, env, method="discrete"):
    k = len(basis_functions)
    #A = np.zeros((k,k)), this might not have an inverse, use the next line instead
    A = np.identity(k) * 0.01
    b = np.zeros(k)
    
    #samples[np.random.choice(len(samples), 100, replace=False)]
    
    for s, a, r, sp in samples:
        phi = _compute_phi(s,a, basis_functions)
        phi_p = _compute_phi(sp, get_policy_action(sp, w, basis_functions, env, method), basis_functions)

        A += np.outer(phi, (phi - gamma*phi_p))
        b = b + phi*r
    
    
    w = np.dot(np.linalg.inv(A),b)
    return w

def _LSTDQ_OPT(samples, basis_functions, gamma, w, env, sigma=0.1, method = "discrete" ):
    k = len(basis_functions)
    B = np.identity(k) * float(1/sigma)
    b = np.zeros(k)
    
    for s, a, r, sp in samples:
        phi = _compute_phi(s, a, basis_functions)
        phi_p = _compute_phi(sp, get_policy_action(sp, w, basis_functions, env, method), basis_functions)

        Bphi = np.dot(B, phi)
        phi_t = (phi - gamma*phi_p).T
        

        top = np.dot(np.outer(Bphi, phi_t), B)
        bottom = 1 + np.dot(phi_t, Bphi)
        B = B - top/bottom
        
        b = b + phi*r
    
    w = np.dot(B, b)

    return w
       

def _compute_phi(s, a, basis_functions):
    phi = np.array([bf(s, a) for bf in basis_functions])
    return phi
    
    

    
def _get_policy_action_discrete(s, w, basis_functions, env):
    a_max = None
    max_score = float("-inf")
    action_space = [0, 1]

    for a in action_space:
        score = np.dot(_compute_phi(s, a, basis_functions), w)
        if score > max_score:
            max_score = score
            a_max = a

    return a_max    
    
    
def _get_policy_actions_continuous_discretized(s, w, basis_functions, env, n_discretizations=10):
    
    a_max = None
    max_score = float("-inf")

    action_space = np.linspace(env.action_space.low[0], env.action_space.high[0], n_discretizations)

    for a in action_space:
        score = np.dot(_compute_phi(s, a, basis_functions), w)

        if score > max_score:
            max_score = score
            a_max = a
            
    return a_max



def _get_policy_actions_continuous(s,w,basis_functions, env):
    f = lambda a: np.dot(_compute_phi(s, a, basis_functions), w)
    x0 = 0
    result = minimize(f, x0, method='L-BFGS-B', options={'xtol': 1e-8, 'disp': True}, bounds = [(-1,1)])
    return result.x

    
def _generate_samples(env, n_samples, n_steps=100):
    samples = []
    print (env.reset())
    for i in range(n_samples):
        env.reset()
        for j in range(n_steps):
            s= env.env.state
            a = env.action_space.sample()
            
            sp,r, _,_ = env.step(a)
            
            sample = (s, a, r, sp)
            samples.append(sample)

    return np.array(samples)