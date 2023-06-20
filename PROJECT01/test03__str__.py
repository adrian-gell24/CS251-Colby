'''test03__str__.py
Test Data class __str__ method for printing `Data` objects
CS 251 Data Analysis and Visualization
Spring 2021
Oliver Layton, Caitrin Eaton, Hannen Wolfe, Stephanie Taylor
'''
import numpy as np

from data import Data


def print_iris(iris_filename):
    iris_data = Data(iris_filename)
    print(iris_data)

    template_str = '''
-------------------------------
data/iris.csv (150x4)
Headers:
  sepal_length    sepal_width    petal_length    petal_width
-------------------------------
Showing first 5/150 rows.
5.1    3.5    1.4    0.2
4.9    3.0    1.4    0.2
4.7    3.2    1.3    0.2
4.6    3.1    1.5    0.2
5.0    3.6    1.4    0.2

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_anscombe(ans_filename):
    ans_data = Data(ans_filename)
    print(ans_data)

    template_str = '''
-------------------------------
data/anscombe.csv (44x2)
Headers:
  x       y
-------------------------------
Showing first 5/44 rows.
10.0    8.04
8.0     6.95
13.0    7.58
9.0     8.81
11.0    8.33

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_spaces(test_filename):
    test_data = Data(test_filename)
    print(test_data)

    template_str = '''
-------------------------------
data/test_data_spaces.csv (3x4)
Headers:
  headers spaces  bad     places
-------------------------------
1.0     2.0     3.0     4.0
5.0     6.0     7.0     8.0
9.0     10.0    11.0    12.0

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_complex(test_filename):
    test_data = Data(test_filename)
    print(test_data)

    template_str = '''
-------------------------------
data/test_data_complex.csv (15x1)
Headers:
  numberstuff
-------------------------------
Showing first 5/15 rows.
4.0
3.0
2.0
1.0
5.0

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (Print Iris data)...')
    print('---------------------------------------------')
    data_file = 'data/iris.csv'
    print_iris(data_file)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Begining test 2 (Print Anscombe data)...')
    print('---------------------------------------------')
    data_file = 'data/anscombe.csv'
    print_anscombe(data_file)
    print('---------------------------------------------')
    print('Finished test 2!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Begining test 3 (Print spaces test data)...')
    print('---------------------------------------------')
    data_file = 'data/test_data_spaces.csv'
    print_spaces(data_file)
    print('---------------------------------------------')
    print('Finished test 3!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Begining test 4 (Print complex test data)...')
    print('---------------------------------------------')
    data_file = 'data/test_data_complex.csv'
    print_complex(data_file)
    print('---------------------------------------------')
    print('Finished test 4!')
    print('---------------------------------------------------------------------------------------')
