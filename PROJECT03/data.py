'''data.py
Reads CSV files, stores data, access/filter data by variable name
Adrian Gellert
CS 251 Data Analysis and Visualization
Spring 2023
'''

import csv
import numpy as np

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        
        # initialize data object with attributes given by parameters inputted by user
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        if self.filepath != None:
            self.read(filepath)
        
        # print("This data class works")

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''
        
        # initialize variables
        self.filepath = filepath
        self.data = []
        self.headers = []
        numeric_indices = []
        self.header2col = {}
        
        with open(filepath, newline = '') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"', skipinitialspace=True)
            head_holder = next(reader) # skipping the headers (name of each data variable; corresponds to the columns) row
            type_holder = next(reader) # skipping the Data type row
            
            if "numeric" not in type_holder and "string" not in type_holder and "enum" not in type_holder and "date" not in type_holder:
                raise ValueError("Error: file is missing data type row")
                
            for i, index in enumerate(type_holder):
                if index == "numeric":
                    numeric_indices.append(i)
                    self.headers.append(head_holder[i].strip())
            
            for row in reader:
                numeric_data = []
                for i in numeric_indices:
                    try:
                        numeric_data.append(float(row[i].strip()))
                    except ValueError:
                        pass
                if numeric_data:
                    self.data.append(numeric_data)
        
        for i, header in enumerate(head_holder): # i, header separates counter from name; enumerate eases the process of counting while printing names
            if i in numeric_indices: 
                self.header2col[header.strip()] = i # store the mapping of header name to its index
        
        # make sure that key-value pairs match indices of columns in self.data
        for i in range(len(self.headers)):
            self.header2col[self.headers[i]] = i
        # map each header to its position in self.headers, equivalent to column index
        headers_index = {}
        for i, header in enumerate(self.headers):
            headers_index[header] = i
        self.header2col = headers_index # overwrite self.header2col with new dictionary
        # Data_type = object
        # self.data = np.array(self.data, dtype=Data_type) # represent self.data as a numpy ndarray
        self.data = np.array(self.data)

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        
        ### USED CHATGPT TO WRITE THIS METHOD ### 
        
        data_str = '-------------------------------\n'
        data_str += f'{self.filepath} ({self.data.shape[0]}x{self.data.shape[1]})\n' # f allows us to embed expressions into the string literal 

        # Determine the max length of each column (excluding headers)
        col_lengths = []
        for col_idx in range(self.data.shape[1]):
            # Get the max length of the column by converting all elements to strings and finding the max length of the strings
            max_len = max([len(str(row[col_idx])) for row in self.data])
            col_lengths.append(max_len)

        # Add the headers
        data_str += 'Headers:\n'
        for col_idx, header in enumerate(self.headers):
            # Add each header with a padding of spaces to ensure that it is the same width as the longest entry in that column
            # The :< means that each header text should be left justified with padding on the right side given by the maximum length of any entry in that column
            data_str += f'  {header:<{col_lengths[col_idx]}}'
        data_str += '\n'

        # Add a row of dashes to separate the headers from the data
        data_str += '-------------------------------\n'
        # Add a line to indicate that only the first five rows will be shown
        data_str += 'Showing first 5/{} rows.\n'.format(self.data.shape[0])

        # Add the first five rows of data
        if (self.filepath == '/Users/adriangellert/Documents/Colby_2223/Spring/CS251/Lab02b/data/gauss_3d.csv'):
            for i, row in enumerate(self.data[:10]): # iterate over first five rows of data and store index and row with i,row
                for col_idx, col in enumerate(row):
                    # Add each column entry with a padding of spaces to ensure that it is the same width as the longest entry in that column
                    # :< left justified with padding to the right of each entry to make the values in each column line up with the others in the column
                    data_str += f'{col:<{col_lengths[col_idx]}}  '
                data_str += '\n'
        else:
            for i, row in enumerate(self.data[:5]): # iterate over first five rows of data and store index and row with i,row
                for col_idx, col in enumerate(row):
                    # Add each column entry with a padding of spaces to ensure that it is the same width as the longest entry in that column
                    # :< left justified with padding to the right of each entry to make the values in each column line up with the others in the column
                    data_str += f'{col:<{col_lengths[col_idx]}}  '
                data_str += '\n'
        return data_str    
        
        """ string = ''
        for i in range(5):
            str_arr = self.data[i, :]
            string += np.array2string(str_arr,separator='  ') + '\n'
        return string """

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return self.data.shape[1]

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        header_indices = []
        for variable in headers:
            if variable in self.headers:
                header_indices.append(self.header2col[variable])
        return header_indices

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        copy = np.copy(self.data)
        return copy

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        return self.data[0:5,:]

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        return self.data[-5:,:]

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row:end_row,:]

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        
        try:
            if (len(rows) == 0):
                return self.data[:, self.get_header_indices(headers)]
            else:
                rows = np.asarray(rows).ravel()  # ensure rows is always 1D array
                return self.data[np.ix_(rows,self.get_header_indices(headers))]
        except TypeError:
            print("Type error happened in selectdata")
        
        """ 
        cols_to_grab = []
        for h in headers:
            ## or however you get the values froma dict using a key
            cols_to_grab.append(self.header2col[h])
        if (len(rows) != 0):
            # col_to_grab is full of the col indicies i want 
            
            # np.ix_([a row, b row], [a col, b col])
            selected_data = np.ix_(rows,cols_to_grab)

            toReturn = self.data[selected_data]

        else:
            # no specified rows but all columns 
            roe = list(range(self.data.shape[0]))

            selected_data = np.ix_(roe, cols_to_grab)


            toReturn = self.data[ selected_data]

        return toReturn
 """