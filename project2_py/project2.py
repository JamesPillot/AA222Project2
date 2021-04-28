#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

def local_descent(x, alpha, d):
    x_next = x + alpha*d
    return x_next

def gradient_descent(x, alpha, g):
    g = g(x)
    d = -g/np.linalg.norm(g)
    x_next = local_descent(x, alpha, d)
    return x_next

def backtrack_line_search_for_grad(f, g, x, d, alpha, p = .5, B = 1e-4):
    y = f(x)
    while(f(x + alpha*d) > y + B*alpha*(np.dot(g,d))):
        alpha *= p
    return alpha

def gradient_descent_with_line_search(f, x, alpha, g, c_prime, pro):
    gr = g(x)
    d = -gr/np.linalg.norm(gr)
    alpha = backtrack_line_search_for_grad(f, gr, x, d, alpha)
    x_next = local_descent(x, alpha, d)
    return x_next

def momentum(g, x, v, alpha, B): 
    g = g(x)
    norm_g = g/np.linalg.norm(g)
    v_next = B*v - alpha*norm_g
    x_next = x + v_next
    return x_next, v_next

def nesterov_momentum(g, x, v, alpha, B, c_prime, pro):
    gr = g(x + B*v)
    norm_gr = gr/np.linalg.norm(gr)
    v_next = B*v - alpha*norm_gr
    x_next = x + v_next
    return x_next, v_next

def p_inv_barrier(x,c):
    return -sum(1/c(x))

def p_quadratic(x, c):
    c_eval = c(x)
    zer_vec = np.zeros(len(c_eval))
    max_vec = np.maximum(c_eval, zer_vec)
    p = (np.linalg.norm(max_vec)**2)
    return p

def p_count(c_eval, h_eval):
    p = np.sum((c_eval > 0)) 
    return p

def basis(i,n):
    return [1.0 if k == i else 0.0 for k in range(n)]

# def Powell_Method(f, x , powell_eps, alpha):
#     n = len(x)
#     U = np.array([basis(i,n) for i in range(n)])
#     delt = np.inf
#     while (delt > powell_eps):
#         x_prime = x
#         for i in range(n):
#             d = U[i]
#             x_prime = x + alpha*d
#         for i in range((n-1)):
#             U[i] = U[i+1]
#         d = x_prime - x
#         U[n-1] = d
#         x_prime = x + alpha*d
#         delt = np.linalg.norm((x_prime - x))
#         x = x_prime
#     return x

def Hooke_Jeeves_penalty(f, c, p, pro, x, alpha, hooke_epsilon, gamma = .5):
    y = f(x) + pro*p(x, c)
    n = len(x)
    while alpha > hooke_epsilon:
        improved = False
        x_best, y_best = x, y
        for i in range(n):
            for sgn in [-1, 1]:
                x_prime = x + sgn*alpha*np.array(basis(i,n))
                y_prime = f(x_prime) + pro*p(x_prime, c)
                if(y_prime < y_best):
                    x_best, y_best, improved = x_prime, y_prime, True
        
        x, y = x_best, y_best
        if (not improved):
            alpha *= gamma
    
    return x



    


def optimize(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    x_last = x0
    v_last = np.zeros(len(x0))
    pro = 1
    gamma = 2
    delta = np.inf
    eps = .01
    powell_eps = .1
    hooke_eps = .1

    while(count() < n):
        if prob == "simple1":
           alpha = .5
           x_best = Hooke_Jeeves_penalty(f, c, p_quadratic, pro, x_last, alpha, hooke_eps)
           pro *= gamma
           if p_quadratic(x_best, c) == 0:
               return x_best

        elif prob == "simple2":
            alpha = .5
            x_best = Hooke_Jeeves_penalty(f, c, p_quadratic, pro, x_last, alpha, hooke_eps)
            pro *= gamma
            if p_quadratic(x_best, c) == 0:
                return x_best
        elif prob == "simple3":
            alpha = .5
            gamma = 8
            x_best = Hooke_Jeeves_penalty(f, c, p_quadratic, pro, x_last, alpha, hooke_eps)
            pro *= gamma
            if p_quadratic(x_best, c) == 0:
                return x_best
        elif prob == "secret1":
            alpha = .5
            gamma = 8
            x_best = Hooke_Jeeves_penalty(f, c, p_quadratic, pro, x_last, alpha, hooke_eps)
            pro *= 8
            if p_quadratic(x_best, c) == 0:
                return x_best

        elif prob == "secret2": 
            alpha = .5
            gamma = 8
            while(delta > eps):
                x_best = Hooke_Jeeves_penalty(f, c, p_inv_barrier, 1/pro, x_last, alpha, hooke_eps)
                delta = np.linalg.norm(x_best - x_last)
                x_last = x_best
                pro *= gamma
            return x_best
        else:
            return float("nan")
            
    return x_best