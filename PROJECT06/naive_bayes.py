'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Adrian Gellert
CS 251/2: Data Analysis Visualization
Spring 2023
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`.
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        self.num_classes = num_classes

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None

        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.class_priors

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.class_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''
        n_samples = data.shape[0]
        
        # summate how many samples correspond to each class
        class_counts = [np.sum(y == i) for i in range(self.num_classes)]
        # divide counts based on class by total number of samples
        self.class_priors = [count / n_samples for count in class_counts]
        
        # separates the data into a list containing different arrays based on the class
        class_data = [data[y == i] for i in range(self.num_classes)]
        n_features = data.shape[1]

        # create a numpy array that has the correct end shape for class_likelihoods
        # (num_classes, num_features)
        self.class_likelihoods = np.zeros((self.num_classes, data.shape[1]))

        for i in range(self.num_classes):
            # Sum the feature counts for all training samples in class i
            feature_counts = np.sum(class_data[i], axis=0)
            words_by_feature = np.sum(feature_counts)
            
            # Compute the class likelihoods for each feature
            # Add 1 to each feature count to avoid zero counts (Laplace smoothing)
            # Divide by the sum of all feature counts + data.shape[1] to get the probability of each feature given the class i
            self.class_likelihoods[i] = (feature_counts + 1) / (words_by_feature + n_features)

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        log_priors = np.log(self.class_priors)
        log_likelihoods = np.log(self.class_likelihoods).T
        
        post_log = log_priors + data @ log_likelihoods
        
        # Predict class with highest log(posterior) probability for each test sample
        predicted_classes = np.argmax(post_log, axis=1)
        return predicted_classes
    
    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        return np.sum(y_pred == y) / len(y)

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        mx = np.zeros((self.num_classes, self.num_classes))
        # print(self.num_classes)
        for true_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                mx[true_class, pred_class] = np.sum((y == true_class) & (y_pred == pred_class))
        return mx
        
