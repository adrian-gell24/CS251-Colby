'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
ADRIAN GELLERT
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        
        super().__init__(data)
        self.orig_dataset = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        orig_data = self.orig_dataset.select_data(headers)
        
        header2col = {}
        for i, head in enumerate(headers):
            header2col[head.strip()] = i
            
        self.data = data.Data(headers=headers, data = orig_data, header2col= header2col)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        
        ones = np.ones((self.data.get_all_data().shape[0],1))
        # print(self.data.get_all_data())
        hom_data = np.hstack((self.data.get_all_data(), ones))
        return hom_data

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        T = np.eye(len(magnitudes) + 1)
        T[:-1,-1] = magnitudes 
        return T

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        S = np.eye(len(magnitudes) + 1)
        S[np.arange(len(magnitudes)), np.arange(len(magnitudes))] = magnitudes
        return S

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        
        Dh = self.get_data_homogeneous()
        # print(Dh.shape)
        Q = self.translation_matrix(magnitudes)
        # print(Q.shape)
        MT = (Q @ Dh.T).T
        MT = MT[:,:-1]
        self.data = data.Data(headers=headers,data=MT,header2col=header2col)
        return MT

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        
        Dh = self.get_data_homogeneous()
        # print(Dh.shape)
        S = self.scale_matrix(magnitudes)
        # print(Q.shape)
        MS = (S @ Dh.T).T
        MS = MS[:,:-1]
        self.data = data.Data(headers=headers,data=MS,header2col=header2col)
        return MS

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        
        Dh = self.get_data_homogeneous()
        MC = (C@Dh.T).T
        MC = MC[:,:-1]
        self.data = data.Data(headers=headers,data=MC,header2col=header2col)
        return MC

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        
        data_set = self.orig_dataset.select_data(self.data.get_headers())
        min, max = data_set.min(), data_set.max()
        denominator = 1/max-min
        magnitudes = len(self.data.get_headers())
        
        scale = self.scale_matrix(magnitudes=[denominator]*magnitudes) # multiplication adds denominator into list a number of times given by magnitudes
        translate = self.translation_matrix(magnitudes=[-min]*magnitudes)
        matrix = (scale@translate)
        
        normalize = self.transform(matrix)
        self.data = data.Data(headers=headers,data=normalize,header2col=header2col)
        return normalize

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        
        data_set = self.orig_dataset.select_data(self.data.get_headers())
        mins, maxs = data_set.min(axis = 0), data_set.max(axis = 0)
        
        denominators = []
        
        for (i,j) in zip(mins,maxs):
            denominators.append(1/(j-i))
        
        scale = self.scale_matrix(denominators) # multiplication adds denominator into list a number of times given by magnitudes
        translate = self.translation_matrix([-val for val in mins])
        matrix = (scale@translate)
        
        normalize = self.transform(matrix)
        self.data = data.Data(headers=headers,data=normalize,header2col=header2col)
        return normalize

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        theta = np.radians(degrees)
        c,s = np.cos(theta), np.sin(theta)
        axis = self.data.get_header_indices([header])
        if (axis[0] == 0):
            rotate = np.array([[1, 0, 0, 0],
                               [0, c, -s, 0],
                               [0, s, c, 0],
                               [0, 0, 0, 1]])
            return rotate
        elif (axis[0] == 1):
            rotate = np.array([[c, 0, s, 0],
                               [0, 1, 0, 0],
                               [-s, 0, c, 0],
                               [0, 0, 0, 1]])
            return rotate
        elif (axis[0] == 2):
            rotate = np.array([[c, -s, 0, 0],
                               [s, c, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            return rotate

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        rotate = self.rotation_matrix_3d(header,degrees)
        
        MR = self.transform(rotate)
        
        self.data = data.Data(headers=headers,data=MR,header2col=header2col)
        return MR
        

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        X = self.orig_dataset.select_data([ind_var])
        Y = self.orig_dataset.select_data([dep_var])
        Z = self.orig_dataset.select_data([c_var])
        color_map = palettable.colorbrewer.sequential.Greys_9
        plt.scatter(x=X, y=Y, c=Z, s=75, cmap=color_map.mpl_colormap, edgecolor='grey')
        cbar = plt.colorbar(orientation = "vertical")
        # print(super(Transformation,t).range([ind_var]))
        # plt.xticks(super().range([ind_var]))
        plt.locator_params(axis='x', nbins=2)
        plt.locator_params(axis='y', nbins=7)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        cbar.set_label(label=c_var)
        plt.title(title)
