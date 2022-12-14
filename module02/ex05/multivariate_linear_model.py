"""Multivariate linear model"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # import the two classes
    import sys
    sys.path.insert(1, '../../module01/ex03/')
    from my_linear_regression import MyLinearRegression as mlr_uni
    sys.path.insert(1, '../../module02/ex04/')
    from mylinearregression import MyLinearRegression as mlr_multi

    # read data
    data = pd.read_csv("./spacecraft_data.csv")

    # UNIVARIATE

    # select a feature to predict Sell price
    for feature in data.head(0).iloc[:, :-1]:
        X = np.array(data[[feature]])
        Y = np.array(data[['Sell_price']])

        # change the initial thetas according to the feature
        params = {'Age': {'thetas': [[1000.0], [-1.0]],
                          'colors' : ['darkblue', 'dodgerblue'],
                          'legend': 'x₁: age (in years)'},
                  'Thrust_power':  {'thetas': [[-1.0], [1000.0]],
                          'colors' : ['darkgreen', 'lime'],
                          'legend': 'x₂: thrust power (in 10Km/s)'},
                  'Terameters':  {'thetas': [[780.0], [-1.0]],
                          'colors' : ['darkviolet', 'violet'],
                          'legend': 'x₃: terameters (in Tmeters)'}}

        # create a first model for the 'Age' feature
        myLR_feat = mlr_uni(thetas = params[feature]['thetas'], alpha = 2.5e-5,
                            max_iter = 100000)

        # fit to adjust the thetas
        myLR_feat.fit_(X[:,0].reshape(-1,1), Y)

        # predict with adjusted thetas
        y_hat = myLR_feat.predict_(X[:,0].reshape(-1,1))

        # output a scatter plot
        plt.figure()
        plt.grid()
        plt.title(f'Prediction of spacecrafts sell price with respect to their'
                  f' {feature.lower()}')
        plt.scatter(X, Y, marker='o', color=params[feature]['colors'][0],
                    label=feature)
        plt.scatter(X, y_hat, marker='o', color=params[feature]['colors'][1],
                    label=f'predicted {feature}')
        plt.xlabel(params[feature]['legend'])
        plt.ylabel('y: sell price (in keuros)')
        plt.legend()
        plt.show()

        # calculate the mean squared error
        print(f'Final loss for {feature}: {myLR_feat.mse_(y_hat, Y)}')
        print(f'Final thetas for {feature}:\n{myLR_feat.thetas}\n')

        # Output
        # 55736.86719... > This is a big error :(


    # MULTIVARIATE

    # X is all feature but the one we try to predict
    Y = np.array(data[['Sell_price']])
    X = np.array(data[['Age','Thrust_power','Terameters']])

    features = list(data.head(0))[:-1]

    # create a first model for the 'Age' feature
    myLR = mlr_multi(thetas = [1.0, 1.0, 1.0, 1.0], alpha = 2.5e-5,
                     max_iter = 1000000)

    # fit to adjust the thetas
    myLR.fit_(X, Y)

    # predict with adjusted thetas
    y_hat = myLR.predict_(X)

    # prepare params for plotting
    params = {'Age': {'colors' : ['darkblue', 'dodgerblue'],
                      'legend': 'x₁: age (in years)'},
              'Thrust_power':  {'colors' : ['darkgreen', 'lime'],
                                'legend': 'x₂: thrust power (in 10Km/s)'},
              'Terameters':  {'colors' : ['darkviolet', 'violet'],
                              'legend': 'x₃: terameters (in Tmeters)'}}

    # plotting the predictions
    for indx, feature in enumerate(features):
        plt.figure()
        plt.grid()
        plt.title(f'Prediction of spacecrafts sell price with respect to their'
                  f' {feature.lower()}')
        plt.scatter(X[:, indx], Y, marker='o',
                    color=params[feature]['colors'][0],
                    label=feature)
        plt.scatter(X[:, indx], y_hat, marker='o', s=20,
                    color=params[feature]['colors'][1],
                    label=f'predicted {feature}')
        plt.xlabel(params[feature]['legend'])
        plt.ylabel('y: sell price (in keuros)')
        plt.legend()
        plt.show()

    # calculate the mean squared error
    print(f'Final loss : {myLR.mse_(y_hat, Y)}') # <- should be way better
