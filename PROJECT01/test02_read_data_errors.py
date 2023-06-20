'''test02_read_data_errors.py
Test Data class error handling
CS 251 Data Analysis and Visualization
Spring 2021
Oliver Layton, Caitrin Eaton, Hannen Wolfe, Stephanie Taylor
'''
import numpy as np

from data import Data


def read_data_error(iris_filename):
    iris_data = Data()
    iris_data.read(iris_filename)


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (CSV error handling)...')
    print('Your program SHOULD crash in the following test.\n')
    print('TODO: Figure out why your code should crash here and then create your own error message\n \
        that helps the user identify the problem and what to do to fix it. You should stop the code\n \
        from running in this situation (e.g. throw an error/exception, exit prematurely).')
    print('------------------')
    data_file = 'data/iris_bad.csv'
    read_data_error(data_file)
    print('------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')
