'''test04_get.py
Test Data class get methods
CS 251 Data Analysis and Visualization
Spring 2021
Oliver Layton, Caitrin Eaton, Hannen Wolfe, Stephanie Taylor
'''
import numpy as np

from data import Data


def test_all_get_methods(iris_filename):
    iris_data = Data(iris_filename)

    print(f"Your iris headers are\n{iris_data.get_headers()}\nand should be\n['sepal_length', 'sepal_width', 'petal_length', 'petal_width']\n")
    print(f"Your iris variable mapping is\n{iris_data.get_mappings()}\nand should be\n{{'sepal_length': 0, 'sepal_width': 1, 'petal_length': 2, 'petal_width': 3}}\n")
    print(f'Your data has {iris_data.get_num_samples()} samples and {iris_data.get_num_dims()} variables/dimensions.\nIt should have 150 samples and 4 variables/dimensions.\n')
    print(f'Your 10th sample is\n{iris_data.get_sample(9)}\nand should be \n[4.9 3.1 1.5 0.1]\n')
    print(f'The indices of the headers sepal_width and petal_width are\n{iris_data.get_header_indices(["sepal_width", "petal_width"])}\nand should be \n[1, 3]\n')


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (Test all get methods)...')
    print('---------------------------------------------')
    data_file = 'data/iris.csv'
    test_all_get_methods(data_file)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')
