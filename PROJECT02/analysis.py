'''analysis.py
Run statistical analyses and plot Numpy ndarray data
ADRIAN GELLERT
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        subset = self.data.select_data(headers, rows)
        mins = np.min(a=subset,axis=0)
        return mins

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        subset = self.data.select_data(headers, rows)
        maxs = np.max(a=subset,axis=0)
        return maxs

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        mins,maxs = (self.min(headers,rows), self.max(headers,rows))
        return mins, maxs

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''
        subset = self.data.select_data(headers, rows)
        arr = np.sum(subset, axis=0)
        means = arr / int(subset[:,0].size)
        return means

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        subset = self.data.select_data(headers, rows)
        means = self.mean(headers, rows)
        arr = np.square(subset - means)
        arr = np.sum(arr, axis = 0)
        variances = arr / (subset.shape[0] - 1)
        return np.array(variances)

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        variances = self.var(headers, rows)
        return variances**(1/2)

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        try:
            x_data = self.data.select_data(headers=[ind_var])
            y_data = self.data.select_data(headers=[dep_var])
            plt.scatter(x_data, y_data)
            plt.title(title)
            if (ind_var == 'citric acid'):
                plt.xlabel(f'{ind_var} (g/dm^3)')
                plt.ylabel(f'{dep_var}')
            elif (ind_var == 'residual sugar'):
                plt.xlabel(f'{ind_var} (g/dm^3)')
                plt.ylabel(f'{dep_var} (vol. %)')
            return (x_data, y_data)
        except TypeError:
            print('error')

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column shousld be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        
        """ fig, axes = plt.subplots(len(data_vars),len(data_vars),figsize=fig_sz, sharex = "row", sharey = "col")
        plt.suptitle(title)

        selected = self.data.select_data(data_vars)

        for i,row in enumerate(axes):
            x = selected[:,i]

            for j,c in enumerate(row):
                y = selected[:,j]
                axes[i,j].scatter(x,y)
                if i == len(axes) - 1:
                    c.set_xlabel(data_vars[j], fontsize="xx-small")
                else:
                    axes[i, j].set_xticks([])
                    axes[i, j].set_xticklabels([])
                if j == 0:
                    c.set_ylabel(data_vars[i],fontsize="xx-small")
                else:
                    axes[i, j].set_yticks([])
                    axes[i, j].set_yticklabels([])
        
        return fig, axes """
    
        figure, axes = plt.subplots(nrows=len(data_vars), ncols=len(data_vars), sharex = "col", sharey = "row", figsize=fig_sz)
        
        # loop through rows (y-axis)
        # loop through columns of each row (x-axis)
        # counter incrementation while moving through a row
        # label y-axis only if in first column
        # label x-axis only if in last row
        # counter = 0
        for i in range(len(data_vars)):
            x = self.data.select_data([data_vars[i]])
            for j in range(len(data_vars)):
                # print(type(self.data))
                # print(i,j)
                # print(self.data.select_data([data_vars[i]]).shape)
                # print(self.data.select_data([data_vars[j]]).shape)
                # axes[i,j].scatter(self.data.select_data([data_vars[j]]), self.data.select_data([data_vars[i]]))
                y = self.data.select_data([data_vars[j]])
                axes[i,j].scatter(y, x)
                if i == (len(data_vars) - 1):
                    if ('length' or 'width' in data_vars[j]):
                        axes[i,j].set_xlabel(f'{data_vars[j]} (cm)')
                    else:
                        axes[i,j].set_xlabel(data_vars[j])
                if j == 0:
                    if ('length' or 'width' in data_vars[i]):
                        axes[i,j].set_ylabel(f'{data_vars[i]} (cm)')
                    else:
                        axes[i,j].set_xlabel(data_vars[i])
                # counter += 1
        figure.suptitle(title)
        plt.tight_layout()
        
        return (figure, axes)
        
