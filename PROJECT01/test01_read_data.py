'''test01_read_data.py
Test Data class constructor and `read` method
CS 251 Data Analysis and Visualization
Spring 2021
Oliver Layton, Caitrin Eaton, Hannen Wolfe, Stephanie Taylor
'''
import numpy as np

from data import Data


def read_data_constructor(iris_filename):
    iris_data = Data(iris_filename)

    print(f'Your file path is:\n  {iris_data.filepath}\nand should be:\n  data/iris.csv\n')
    print(f"Your iris headers are\n  {iris_data.headers}\nand should be\n  ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n")
    print(f"Your iris variable mapping is\n  {iris_data.header2col}\nand should be\n  {{'sepal_length': 0, 'sepal_width': 1, 'petal_length': 2, 'petal_width': 3}}\n")
    print(f'Your data is a ndarray? {isinstance(iris_data.data, np.ndarray)}')
    print(f'Your data has {iris_data.data.shape[0]} samples and {iris_data.data.shape[1]} variables/dimensions.\nIt should have 150 samples and 4 variables/dimensions.')


def read_data_separately(iris_filename):
    iris_filename = 'data/iris.csv'
    iris_data = Data()
    print('Before calling read...')
    print(f"Your iris headers are:\n  {iris_data.headers}\nand should be:\n  None (or []).\n")

    iris_data.read(iris_filename)

    print('After calling read...')
    print(f'Your file path is:\n  {iris_data.filepath} and should be:\n  data/iris.csv\n')
    print(f"Your iris headers are\n  {iris_data.headers}\nand should be\n  ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n")
    print(f"Your iris variable mapping is\n  {iris_data.header2col}\nand should be\n  {{'sepal_length': 0, 'sepal_width': 1, 'petal_length': 2, 'petal_width': 3}}\n")
    print(f'Your data is a ndarray? {isinstance(iris_data.data, np.ndarray)}')
    print(f'Your data has {iris_data.data.shape[0]} samples and {iris_data.data.shape[1]} variables/dimensions.\nIt should have 150 samples and 4 variables/dimensions.')


def read_data_spaces(test_filename):
    test_data = Data(test_filename)
    print(f'Your test data looks like:\n', test_data.data)
    print('You should see:')
    print('Your test data looks like:\n [[ 1.  2.  3.  4.]\n [ 5.  6.  7.  8.]\n [ 9. 10. 11. 12.]]')
    print('Pay attention to the data type! The numbers should be floats (not have quotes around them).')


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (Read data in constructor)...')
    print('---------------------------------------------')
    data_file = 'data/iris.csv'
    read_data_constructor(data_file)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Begining test 2 (Read data separately)...')
    print('---------------------------------------------')
    read_data_separately(data_file)
    print('---------------------------------------------')
    print('Finished test 2!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Begining test 3 (Data with spaces)...')
    print('---------------------------------------------')
    spaces_filename = 'data/test_data_spaces.csv'
    read_data_spaces(spaces_filename)
    print('---------------------------------------------')
    print('Finished test 3!')
    print('---------------------------------------------------------------------------------------')
