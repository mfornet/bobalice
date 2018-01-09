"""
    SFAlice
    Stochastic Finite Automata - Alice
    ==================================

    Alice will try to fit an stochastic automata to predict the hidden language.

    See `eval_word` method to understand how alice detect if a word belong to
    the language or not.

    ## Notes:
        The total number of states needs to be specified manually. 
        While a large number of statesis hard to optimize a small number 
        of states might not fit well the automata.

    ## Ideas to improve results:
        +   Make better questions. With a better training set we might
            expect better performance. While making question run small
            optimization, and find words that has higher uncertainty to
            make more accurate answers.

        +   Loss function:
                Try with other loss functions to achieve better score.
                Try with other optimizers. Maybe some metahauristics can perform
                good here.
"""

import sys, io

import numpy as np
import scipy.optimize as opt
import sklearn.cross_validation as cv

### Helpers

def random_word(alphabet, p=0.5):
    alphabet = list(alphabet)
    word = ""
    while np.random.random() < p:
        word += np.random.choice(alphabet)
    return word

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ravel(transition, terminal):
    return np.append(transition.ravel(), terminal)

def unravel(theta, alpha_size, states):
    transition = np.reshape(theta[:alpha_size * states * states], 
                        (alpha_size, states, states))
    terminal = theta[-states:]
    return transition, terminal

class BagOfWord(dict):
    def __init__(self, alphabet):
        super().__init__()
        self.alphabet = alphabet

    def random(self):
        while True:
            word = random_word(self.alphabet)
            if not word in self:
                break

        self[word] = None
        return word

    def get_train(self):
        train = []
        for word, lang in self.items():
            pos = [self.alphabet.index(c) for c in word]
            train.append((pos, lang))
        return train

def interpret_sfa(transition, terminal, alphabet):
    alpha_size = len(alphabet)
    states = terminal.shape[0]

    print("States:", states)
    print("Alphabet:", alphabet)

    print(transition)
    print(terminal)

    next_state = transition.argmax(axis=1)
    r_terminal = terminal > 0

    for i in range(states):
        print("State {} : {}".format(i, 'terminal' if r_terminal[i] else 'not terminal'))

        for idx, c in enumerate(alphabet):
            print("Reading {} go to: {}".format(c, next_state[idx][i]))
 
### Evaluation

def eval_word(transition, terminal, word, get_uncertainty=False):
    """
    params:
        transition:
        terminal:
        word:
        get_uncertainty:

    return: (p, u)
        p:  Probability that the word belongs to the language
            1 belong, 0 otherwise
        u:  Sum of the uncertainty among all probabilities 
            (only if `get_uncertainty` is True)
    """

    # List of probility of being in a particular state
    pi = np.zeros(terminal.shape)
    pi[0] = 1.

    uncertainty = 0.

    for p in word:
        pi = transition[p] @ pi
        pi = sigmoid(pi)
        pi /= pi.sum()

        uncertainty += np.sqrt(pi * (1. - pi)).sum()

    p = sigmoid(terminal @ pi)

    if not get_uncertainty:
        return p
    else:
        return p, uncertainty

def score(theta, X, y, alpha_size, states):
    """
    Predict the outcome given the SFA parameters
    and compute the accuracy using test data

    params:
        theta:
        X:
        y:
        alpha_size:
        states:

    return:
        accuracy on test data (float)
    """
    trans, term = unravel(theta, alpha_size, states)

    ok = 0.
    count = 0.

    for Xi, yi in zip(X, y):
        y_pred = eval_word(trans, term, Xi)
        y_class = int(y_pred > .5)

        count += 1
        if y_class == yi:
            ok += 1

    return ok / count

### Error

def get_error(X, y, alpha_size, states, kappa, omega):
    """
    Metafunction - Create the function to optimize
    given hyperparameters and datasets. 

    The intention is to use an optimizer on
    the returned function

    params:
        X: list of list representing each word
        y: classification of each word
        alpha_size: Size of the alphabet
        states: Number of states in the representation

        kappa: Regularization parameter to control weights
        omega: Regularization parameter to increase confidence

    return:
        function to minimize
    """

    def error(theta):
        trans, term = unravel(theta, alpha_size, states)

        real_error = 0.
        count = 0.

        total_uncertainty = 0.
        total_variables = 0.

        for Xi, yi in zip(X, y):
            y_pred, uncertainty = eval_word(trans, term, Xi, get_uncertainty=True)

            total_uncertainty += uncertainty
            total_variables = len(Xi) * states

            if yi == 1:
                real_error -= np.log(y_pred)
            else:
                real_error -= np.log(1. - y_pred)

            count += 1

        real_error /= count

        # Regularization
        weight_control = (theta * theta).sum() * kappa / len(theta)
        confidence_control = total_uncertainty / total_variables * omega

        # print("Distance:", real_error)
        # print("Weight control:", weight_control)
        # print("Confidence control:", confidence_control)

        return real_error + weight_control + confidence_control

    return error

### Constants

# Show information
SHOW_INFO = True

# Number of states of the stochastic finite automata
STATES = 2

# Number of training epochs
EPOCHS = 3

# Regularization parameter
KAPPA = 0.001
OMEGA = 0.01

# Maximum number of function evaluation during optimization
MAX_EVAL = None

### Code

def main():
    # Redirect file descriptors

    bob = sys.stdout
    devnull = io.StringIO()
    sys.stdout = sys.stderr if SHOW_INFO else devnull

    # Begin communication

    n = int(input())
    alphabet = input()

    # Phase 1
    # Random questions

    bag = BagOfWord(alphabet)

    for i in range(n):
        # Get and ask a random word
        word = bag.random()
        print(word, flush=True, file=bob)
        value = input()
        value = 1 if value == 'yes' else 0

        # and update its value
        bag[word] = value

    # Preprocess training data

    train = bag.get_train()
    X, y = [], []

    for Xi, yi in train:
        X.append(Xi)
        y.append(yi)

    # Initialize constants

    states = STATES
    kappa = KAPPA
    omega = OMEGA
    alpha_size = len(alphabet)

    total_variables = alpha_size * states * states + states

    # Train

    best_theta = None
    best_error = None
    best_score = None

    for i in range(EPOCHS):
        print("\nEpoch:", i)

        # Random initialize all weights
        theta = np.random.randn(total_variables)

        X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=.2)
        error = get_error(X_train, y_train, alpha_size, states, kappa, omega)

        print("Train size:", len(X_train))
        print("Test size:", len(X_test))

        print("Start training...")

        # Optimize using Nelder and Mead
        theta, error_train, iterations, evaluations, _ = opt.fmin(error, theta, 
                                                                    maxfun=MAX_EVAL, 
                                                                    full_output=True)

        score_train = score(theta, X_train, y_train, alpha_size, states)

        error_test = get_error(X_test, y_test, alpha_size, states, kappa, omega)(theta)
        score_test = score(theta, X_test, y_test, alpha_size, states)

        print("Iterations:", iterations)
        print("Evaluations:", evaluations)

        print("Error on train:", error_train)
        print("Score on train:", score_train)

        print("Error on test:", error_test)
        print("Score on test:", score_test)

        # Update best theta
        if best_theta is None or score_test > best_score or \
            (score_test == best_score and error_test < best_error):

            best_theta = theta
            best_score = score_test
            best_error = error_test

    # Build best model

    transition, terminal = unravel(best_theta, alpha_size, states)

    interpret_sfa(transition, terminal, alphabet)

    # Phase 2
    # Guess questions

    for i in range(n):
        word = input()
        pos = [alphabet.index(c) for c in word]

        y_pred = eval_word(transition, terminal, pos)

        ok = int(y_pred > .5)
        print("yes" if ok == 1 else "no", flush=True, file=bob)


if __name__ == '__main__':
    main()