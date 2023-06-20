'''linear_regression.py
Subclass of Analysis that performs linear regression on data
ADRIAN GELLERT
CS251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis
import data

class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(headers=ind_vars)
        self.y = self.data.select_data(headers=[dep_var])
        Ahat = np.hstack((np.ones((self.A.shape[0], 1)), self.A))
        c, rs, rank, _ = scipy.linalg.lstsq(Ahat, self.y)
        self.slope = c[1:,]
        self.intercept = float(c[0])
        self.R2 = self.r_squared(self.predict())
        self.residuals = self.compute_residuals(self.predict())
        self.mse = self.compute_mse()

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        # X = np.atleast_2d(X)
        # if self.p > 1:
        #     if X is None:
        #         self.A = self.make_polynomial_matrix(self.A, self.p)
        #     else:
        #         self.A = self.make_polynomial_matrix(X, self.p)
            
        
        if X is None:
            # print(f'slope: {self.slope}')
            # X = self.A
            # print(f'X: {X.shape}')
            # print(self.A)
            # print(f'A: {self.A.shape}')
            # print(f'intercept: {self.intercept.shape}')
            y_pred = (self.A @ self.slope) + self.intercept
        else:
            # print(f'X.shape: {X.shape}')
            # print(f'self.A.shape: {self.A.shape}')
            # print(f'self.slope.shape: {self.slope.shape}')
            # print(f'self.intercept: {self.intercept}')
            # y_pred = (X @ self.slope.reshape((-1, 1))) + np.array([self.intercept]).reshape(1, -1)
            # print(f'slope: {self.slope}')
            # print(f'X: {X.shape}')
            # print(f'intercept: {self.intercept.shape}')
            y_pred = X @ self.slope + self.intercept
        
        # print(type(y_pred))
        return y_pred
        
        # if X.shape[1] == self.A.shape[1]:
        #     X = self.A
        # else:
        #     X = self.make_polynomial_matrix(X, p=self.p)

        # if self.slope is None:
        #     print('Error: linear regression has not been trained.')
        #     return None

        # y_pred = X @ self.slope + self.intercept
        # return y_pred
    
    

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        E = np.sum(self.compute_residuals(y_pred)**2)
        S = np.sum((self.y - self.y.mean())**2)
        R2 = float(1 - E/S)
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        residuals = self.y - y_pred
        return residuals

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        y_pred = self.predict(self.A)
        residuals = self.compute_residuals(y_pred)
        n = self.y.shape[0]
        return float(1/n * sum(residuals**2))

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        
        x,y = super().scatter(ind_var, dep_var, title)
        # plt.scatter(x,y)
        
        line_x = np.linspace(x.min(), x.max(), x.shape[0]).reshape((x.shape[0],1))
        line_x_polynomial = np.reshape(line_x, (line_x.shape[0],1))
        if self.p > 1:
            line_x_polynomial = self.make_polynomial_matrix(line_x, self.p)
        # self.A = self.make_polynomial_matrix(line_x, p)
        # line_y = self.predict(line_x)
        regression = np.squeeze(line_x_polynomial @ self.slope + self.intercept)
        
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.plot(line_x, regression, color='red', label='Regression Line')
        # print(y)
        plt.title((title + '\nR^2 = ' + str(self.r_squared(self.predict())) + '\nMSSE = ' + str(self.mse)), fontsize = 10)
        # plt.tight_layout()
        
    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        
        fig, axes = super().pair_plot(data_vars, fig_sz=fig_sz, title=None)
                    
        for i, row in enumerate(axes):
            dependent = data_vars[i]
            for j, col in enumerate(row):
                independent = data_vars[j]
                self.linear_regression([independent], dependent)
                
                line_x = np.linspace(self.A.min(), self.A.max(), self.A.shape[0]).reshape((self.A.shape[0],1))
                line_y = self.predict(line_x)
                
                axes[i,j].plot(line_x, line_y, color='red')
                col.set_title(f'R^2 = {self.R2:.2f}')
                
                if i == j:
                    if hists_on_diag == True:
                        numVars = len(data_vars)
                        axes[i, j].remove()
                        axes[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
                        if j < numVars-1:
                            axes[i, j].set_xticks([])
                        else:
                            axes[i, j].set_xlabel(data_vars[i])
                        if i > 0:
                            axes[i, j].set_yticks([])
                        else:
                            axes[i, j].set_ylabel(data_vars[i])
                        axes[i, j].hist(self.A, bins = self.A.shape[0])
                        axes[i, j].set_title(f'R^2 = {self.R2:.2f}')
        plt.tight_layout()
        
        
            

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        
        matrix = np.ones((A.shape[0], p))
        for i in range(1, p + 1):
            matrix[:, i - 1] = np.squeeze(A**i)
        # matrix = np.squeeze(matrix)
        # print(f'Matrix shape: {matrix.shape}')
        return matrix

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_var = ind_var
        self.dep_var = dep_var
        self.p = p
        # print(self.ind_var, self.dep_var)
        # print(type(self.data))
        self.A = self.make_polynomial_matrix(self.data.select_data(headers=ind_var), p = self.p)
        self.y = self.data.select_data(headers=[dep_var])
        Ahat = np.hstack((np.ones((self.A.shape[0], 1)), self.A))
        # print(f'A shape: {self.A.shape}')
        # print(f'y shape: {self.y.shape}')
        # print(f'Ahat shape: {Ahat.shape}')
        c, rs, rank, _ = scipy.linalg.lstsq(Ahat, self.y)
        self.slope = c[1:]
        # print(f'slope: {self.slope}')
        self.slope = self.slope.reshape((self.slope.shape[0], 1))
        # self.intercept = np.float(c[0,0])
        self.intercept = c[0,0]
        # self.intercept = self.intercept.astype(np.float)
        self.R2 = self.r_squared(self.predict())
        self.residuals = self.compute_residuals(self.predict())
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p
        self.A = self.make_polynomial_matrix(self.data.select_data(headers=ind_vars), p = self.p)
        self.y = np.reshape(self.data.data[:, self.data.header2col[dep_var]], (len(self.data.data), 1))
        self.R2 = self.r_squared(self.predict())
        self.residuals = self.compute_residuals(self.predict())
        self.mse = self.compute_mse()
