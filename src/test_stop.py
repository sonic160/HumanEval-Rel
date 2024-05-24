from typing import List
test = check(p_f_interval)

def check(candidate):
        import numpy as np
        def weibull_cdf(x, scale=2, shape=3):
                return 1 - np.exp(-((x / scale) ** shape))
        try:
                assert candidate(lambda x: weibull_cdf(x), 1, .5) == weibull_cdf(1.5) - weibull_cdf(1)
                return 'pass'
        except:
                return 'fail'

test = check(p_f_interval)