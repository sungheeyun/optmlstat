from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    boston = load_boston()

    boston = load_boston()
    boston = load_boston()

    X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(boston.data,
                                                                                    boston.target,
                                                                                    test_size=0.2,
                                                                                    random_state=42)

    robust = SGDRegressor(loss='huber',
                          penalty='l2',
                          alpha=0.0001,
                          fit_intercept=False,
                          max_iter=100,
                          shuffle=True,
                          verbose=1,
                          epsilon=0.1,
                          random_state=42,
                          learning_rate='invscaling',
                          eta0=0.01,
                          power_t=0.5)

    sc_boston = StandardScaler()
    X_train_boston = sc_boston.fit_transform(X_train_boston)
    X_test_boston = sc_boston.transform(X_test_boston)

    robust.fit(X_train_boston, y_train_boston)

    print(mean_squared_error(y_test_boston, robust.predict(X_test_boston)) ** 0.5)


    pass

