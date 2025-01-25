# assetpricing/derivatives/options/binomial.py
import numpy as np

class BinomialTree:
    """
    American/European option pricing using binomial tree.
    """
    def __init__(self, S, K, T, r, sigma, steps, option_type='call', american=True):
        self.dt = T/steps
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1/self.u
        self.p = (np.exp(r*self.dt) - self.d)/(self.u - self.d)
        self.steps = steps
        self.type = option_type
        self.american = american
        self.S = S
        self.K = K
        self.r = r
        
    def price(self):
        # Tree construction
        stock_tree = np.zeros((self.steps+1, self.steps+1))
        stock_tree[0,0] = self.S
        for i in range(1, self.steps+1):
            stock_tree[i,0] = stock_tree[i-1,0] * self.u
            for j in range(1, i+1):
                stock_tree[i,j] = stock_tree[i-1,j-1] * self.d
                
        # Option value calculation
        option_tree = np.zeros_like(stock_tree)
        option_tree[-1] = np.maximum(
            0, stock_tree[-1] - self.K if self.type == 'call' 
            else self.K - stock_tree[-1]
        )
        
        for i in reversed(range(self.steps)):
            for j in range(i+1):
                option_tree[i,j] = np.exp(-self.r*self.dt) * (
                    self.p*option_tree[i+1,j] + (1-self.p)*option_tree[i+1,j+1]
                )
                if self.american:
                    exercise_now = stock_tree[i,j] - self.K if self.type == 'call' else self.K - stock_tree[i,j]
                    option_tree[i,j] = np.maximum(option_tree[i,j], exercise_now)
        
        return option_tree[0,0]