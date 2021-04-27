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

def backtrack_line_search(f, g, x, d, alpha, p = .005, B = 1e-12):
    y = f(x)
    while(f(x + alpha*d) > y + B*alpha*(np.dot(g,d))):
        alpha *= p
    return alpha

def gradient_descent_with_line_search(f, x, alpha, g):
    g = g(x)
    d = -g/np.linalg.norm(g)
    alpha = backtrack_line_search(f, g, x, d, alpha)
    x_next = local_descent(x, alpha, d)
    return x_next

def momentum(g, x, v, alpha, B): 
    g = g(x)
    norm_g = g/np.linalg.norm(g)
    v_next = B*v - alpha*norm_g
    x_next = x + v_next
    return x_next, v_next

def nesterov_momentum(g, x, v, alpha, B, c_prime, pro):
    gr = g(x + B*v) + pro*c_prime
    norm_gr = gr/np.linalg.norm(gr)
    v_next = B*v - alpha*norm_gr
    x_next = x + v_next
    return x_next, v_next

def p_quadratic(c_eval, h_eval):
    zer_vec = np.zeros(len(c_eval))
    max_vec = np.maximum(c_eval, zer_vec)
    p = (np.linalg.norm(max_vec)**2) + (np.linalg.norm(h_eval)**2)
    return p
# def penalty_method_quadratic(f, g, c, jac_c_t, x, min_func, alpha, beta, v, pro, gamma):
#     c_eval = c(x)
#     zer_vec = np.zeros(len(c_eval))
#     c_prime = 2*np.sum(jac_c_t*np.maximum(c_eval, zer_vec))
#     x_next, v_next = min_func(g, x, v, alpha, beta, c_prime, pro)
#     return x_next, v_next
    


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

    while(count() < n):
        if prob == "simple1":
            # Hyperparameters
            alpha = .1222
            B = .55
            gamma = 2

            c_eval = c(x_last)
            p = p_quadratic(c_eval, 0)
            if(p == 0):
                return x_last

            # Gradients and Jacobian of constraints
            c1_grad = np.array([1, 2*x_last[1]])
            c2_grad = np.array([-1, -1])
            jac_c = np.array([c1_grad, c2_grad])
            jac_c_t = np.transpose(jac_c)
            c_eval = c(x_last)
            zer_vec = np.zeros(len(c_eval))
            c_prime = 2*np.sum(jac_c_t*np.maximum(c_eval, zer_vec))
            # Minimize with quadratic penalty
            # breakpoint()

            x_next, v_next = nesterov_momentum(g, x_last, v_last, alpha, B, c_prime, pro)

            # Update pro and x, v values for minimizer (nesterov)
            pro *= gamma
            x_last = x_next
            v_last = v_next
        else:
            return float("nan")
        # elif prob == "simple2": 
        # elif prob == "simple3": 
        # elif prob == "secret1":
           
        # elif prob == "secret2":       
        

    x_best = x_last

    return x_best