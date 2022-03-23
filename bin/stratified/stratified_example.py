
import numpy as np
import strat_models

bm = strat_models.BaseModel(loss=strat_models.logistic_loss(intercept=True),
								reg=strat_models.sum_squares_reg(lambd=1))

K = 100
n = 10
num_samples = 500

X = np.random.randn(num_samples, n)
Z = np.random.randint(K, size=num_samples)
Y = np.random.randn(num_samples, 1)

data = dict(X=X, Y=Y, Z=Z)
