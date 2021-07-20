import random
import math
from sklearn import svm
import sklearn.model_selection
import numpy as np


def create_binary_chroms(num_chroms, length, num_features):
    '''
    Create a number of binary encoded chromosome of a specific length and with the
    a specific number of features

    :param num_chroms; The number of chromosomes to make
    :param length: The length of the chromosome
    :param num_features: The number of ones in the binary chromosome (i.e. the
                         number of features that will be selected)

    :return: A dictionary of N chromosomes of length = length and with num_features
             number of non-zero entries.
    '''

    # For storing the chroms
    chrom_dict = {}

    # Create the chroms
    for i in range(num_chroms):
        chrom = [0]*length

        # Randomly select some places in the list
        randoms = random.sample(range(len(chrom)), k=num_features)
        # Turn these random features on
        for j in range(len(randoms)):
            chrom[randoms[j]] = 1

        # Store the chromosome in the dictionary
        chrom_dict[i] = chrom

    return chrom_dict


def evaluate_chroms_svm(chromosomes, data, labels, folds=10):
    '''
    Uses a Linear SVM to evaluate the performance of the chromosomes based upon
    their classification accuracy. Uses Kfold cross-validation

    :param chromosomes: A dictionary of binary encoded chromosomes with length
                       equal to the number of columns in the dataframe
    :param data: A dataframe of data to be classified
    :param labels: Corresponding labels for classification
    :param folds: How many splits to use in k-fold cross validation

    :return: Returns a dictionary with keys the same as the chromosome dictionary,
             but with accuracy instead of the binary chromosome encoding
    '''
    # Create a new dictionary to store values
    chrom_dict = {}

    # Look through each chromosome, get the features and evaluate an svm on them
    for key in chromosomes.keys():
        # Get the index of the non-zero features
        feature_indexes = [i for i, val in enumerate(chromosomes[key]) if val]

        # Skip this chromosome if there are no features
        if len(feature_indexes) == 0:
            continue

        # Extract only the selected features for training/testing
        trimmed_data = data.iloc[:, feature_indexes]
        trimmed_data_array = trimmed_data.to_numpy()
        labels_array = np.array(labels)

        # Make some folds using sklearn
        kfold = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)

        # Container to store accuracies from the folds
        accuracies = []

        # Go through each fold
        for train_index, test_index in kfold.split(trimmed_data, labels):

            # Split the data based upon the fold's indexes
            x_train = trimmed_data_array[train_index]
            x_test = trimmed_data_array[test_index]
            y_train = labels_array[train_index]
            y_test = labels_array[test_index]

            # Create and fit the classifier
            clf = svm.SVC(kernel='linear', C=0.1)
            clf.fit(x_train, y_train)

            # Predict and add this folds accuracy onto the list of accuracies
            preds = clf.predict(x_test)
            fold_accuracy = sum(preds==y_test)/len(y_test)
            accuracies.append(fold_accuracy)

        # Get the average accuracy for the chromosome
        accuracy = sum(accuracies)/len(accuracies)

        # Add accuracy into the dictionary
        chrom_dict[key] = accuracy

    # Return evaluation dictionary
    return chrom_dict


def select(chromosomes, chromosomes_evaluated, percentage, threshold=0, min_num=3):
    '''

    :param chromosomes: A dictionary of binary encoded chromosomes
    :param chromosomes_evaluated: A dictionary of evaluation scores and keys identifying unique chromosomes
    :param percentage: What percentage of the total number of chromosomes will be selected for the next
                       generation (expressed as a decimal 0.3 = 30%)
    :param threshold: The threshold fitness that a chromosome must exceed to be viable for selection, default
                      is 0 (in practice this measure hasn't been useful)
    :param min_num: The minimum number of chromosomes that are allowed through

    :return: A dictionary of the selected binary encoded chromosomes
    '''

    # The number that survive
    n_survivors = int(len(chromosomes_evaluated.keys())*percentage)

    # Extract the best n chromosome keys and their associated evaluation scores
    chrome_dict = {key: chromosomes_evaluated[key] for key in
                   sorted(chromosomes_evaluated,
                                key=chromosomes_evaluated.get,
                                reverse=True)[:n_survivors]}

    # Check that these values are above the threshold starting with the chromosome
    # with the lowest threshold
    for key in chrome_dict.keys().__reversed__():

        # If we're going to remove too many
        if len(chrome_dict.keys()) <= min_num:
            continue

        # Otherwise, pop the key that has lowest evaluation score and is below the threshold
        else:
            if chrome_dict[key] < threshold:
                chrome_dict.pop(key)

    # Finally, return a new dictionary with the surviving chromosomes in binary format
    survivors = {key: chromosomes[key] for key in chrome_dict.keys()}
    return survivors


# Two point crossover
def binary_crossover(survivors, n_chroms_out=10):
    '''
    Recombines two chromosomes by randomly selecting two points at which to cut the chromosome
    and swapping the middle section. Will start at the best chromosome and work to the worst
    chromosome, i.e. combines chrom1 with chrom2, then chrom 1 with chrom3, then chrom1 with chrom4
    etc... until out of chrom1 combinations then continues down the list. Given N chromosomes in, there
    are 2 * (N choose 2) combinations out, or 2 * (n!)/((n-2)! * 2!)

    :param survivors: A dictionary of binary encoded chromosomes
    :param n_chroms_out: The desired number of output chromosomes

    :return: A dictionary of binary encoded chromosomes
    '''

    # Dictionary for the new chromosomes
    new_chromosomes = {}

    chroms_in = len(survivors.keys())
    # Check to see if there are enough chromosomes to make the required new number
    if n_chroms_out > 2 * (math.factorial(chroms_in)/\
                        (math.factorial((chroms_in-2)) * 2)):
        print("There are not enough surviving chromosomes to make {} new combinations".format(n_chroms_out))
        return

    # Now we just need to figure out how to cut it
    for i in range(len(survivors.keys())):

        # If we're on the final chromosome in the list
        if i == len(survivors.keys()) - 1:
            break

        # Get the fittest chromosone
        fittest = survivors[[j for j in survivors.keys()][i]]

        # Randomly select a 2 points to cut the chromosone on. Sort from lowest to highest.
        cuts = sorted(random.sample(range(len(fittest)), k=2))

        for k in range(i+1, len(survivors.keys())):

            # Look at the next chromosome in the list
            next_chrom = survivors[[j for j in survivors.keys()][k]]

            # Make new chromosomes from the two
            new_chrom_1 = fittest[:cuts[0]] + next_chrom[cuts[0]:cuts[1]] + fittest[cuts[1]:]
            new_chrom_2 = next_chrom[:cuts[0]] + fittest[cuts[0]:cuts[1]] + next_chrom[cuts[1]:]

            # Check to see if we haven't exceeded the limit of new chromosomes and if not
            # add the new chromosome to the dictionary
            if len(new_chromosomes.keys()) >= n_chroms_out:
                return new_chromosomes

            elif len(new_chromosomes.keys()) - n_chroms_out == 1:
                # If there is only room for one new chromosome, add the one with the most material
                # from the fittest chromosome
                new_chromosomes[len(new_chromosomes.keys())] = new_chrom_1


            else:
                # Otherwise add them both
                new_chromosomes[len(new_chromosomes.keys())] = new_chrom_1
                new_chromosomes[len(new_chromosomes.keys())] = new_chrom_2

    # return the new chromosomes
    return new_chromosomes


def mutate(chromosomes, percent_range=(0.005, 0.01)):
    '''
    Randomly select and mutate (change 1 to 0, or 0 to 1) a number of bits in binary encoded
    chromosomes. The number of bits to be mutated is randomly selected from the percent_range according
    to the uniform distribution. I.e, percent_range = (0.05, 0.1) means 5-10% of chromosomes will be mutated

    :param chromosomes: A dictionary of binary encoded chromosomes to mutate. They must all be the same
    length. This function changes the provided dictionary.
    :param percent_range: A tuple indicating the range of bits to be mutated.

    :return: A dictionary of mutated binary encoded chromosomes
    '''

    # Order from lowest to highest and sample the percentage to be mutated
    percent_range = sorted(percent_range)
    percentage = random.uniform(percent_range[0], percent_range[1])

    # Calculate the number to be mutated (rounded down)
    n = int(percentage * len(chromosomes[0]))

    for key in chromosomes:
        # Find the indexes to be mutated
        mutation_index = random.sample(range(len(chromosomes[key])), k=n)

        # Change the bits at the given indexes to the opposite character
        for ind in mutation_index:
            chromosomes[key][ind] = (chromosomes[key][ind] - 1) * -1

    # Return the mutated dictionary (Maybe not strictly necessary as the dictionary has already been changed)
    return chromosomes


def trim(chromosomes, max_features=10):
    '''
    Stochastically trims the number of engaged features (1's) in a binary encoded chromosome to the number specified
    by the max_features argument

    :param chromosomes: A dictionary of binary encoded chromosomes. This dictionary will be changed by this
    function
    :param max_features: The maximum number of features allowed

    :return: A dictionary of chromosomes with a number of features (1's) less than the maximum provided
    '''

    for key in chromosomes:
        if sum(chromosomes[key]) > max_features:

            # Get the indexes of the features
            features = [i for i, element in enumerate(chromosomes[key]) if element != 0]

            # How many need to be removed
            n = len(features) - max_features

            # Randomly sample the indexes of the features to find the ones to remove
            features_to_drop = random.sample(features, k=n)

            # Go through the list of features to drop and swap the 1's to a 0
            for ind in features_to_drop:
                chromosomes[key][ind] = 0

    # Return the dictionary. Perhaps not really necessary because the function changes the underlying dictionary

    return chromosomes


def create_custom_chroms(num_chroms, custom_vals):
    '''
    Create a number of custom chromosomes with length and values drawn from the custom_vals input

    :param num_chroms; The number of chromosomes to make
    :param custom_vals: A list of tuples where each tuple defines a minimum and maximum value for
                        each single gene. Those values will then be used to generate starting values
                        for the EA. Values should be input as round numbers for integers and floats for
                        floats (otherwise float values will be generated for integers).

    :return: A dictionary of N chromosomes of length = length
    '''

    # For storing the chroms
    chrom_dict = {}

    # Create the chroms
    for i in range(num_chroms):
        chrom=[0]*len(custom_vals)

        # Create the custom values
        for j in range(len(chrom)):
            # If the tuple is an integer
            if type(custom_vals[j][0]) is int:
                # Generate a random integer between the values
                chrom[j] = random.randrange(custom_vals[j][0], custom_vals[j][1])

            # If it is a float
            elif type(custom_vals[j][0]) is float:
                # Generate a random float between the values
                chrom[j] = random.uniform(custom_vals[j][0], custom_vals[j][1])

            # If not valid
            else:
                print("Invalid input to create_custom_chroms, must be list of tuples of"
                      " integer or float values")

        # Store the chromosome in the dictionary
        chrom_dict[i] = chrom

    return chrom_dict

def custom_crossover(survivors, n_chroms_out):
    '''
    Creates a number of new chromosomes from the survivors. Simply takes the average of the two

    :param survivors: A dictionary of chromosomes
    :return:
    '''
    # Dictionary for the new chromosomes
    new_chromosomes = {}

    # Now we just need to figure out how to cut it
    for i in range(len(survivors.keys())):

        # If we're on the final chromosome in the list
        if i == len(survivors.keys()) - 1:
            break

        # Get the fittest chromosome
        fittest = survivors[[j for j in survivors.keys()][i]]

        for k in range(i+1, len(survivors.keys())):

            # Look at the next chromosome in the list
            next_chrom = survivors[[j for j in survivors.keys()][k]]

            # Make new chromosomes from the two
            new_chrom = []
            for j in range(len(fittest)):
                if type(fittest[j]) is int:
                    new_chrom.append(int((fittest[j] + next_chrom[j])/2))
                else:
                    new_chrom.append((fittest[j] + next_chrom[j]) / 2)

            # Check to see if we haven't exceeded the limit of new chromosomes and if not
            # add the new chromosome to the dictionary
            if len(new_chromosomes.keys()) >= n_chroms_out:
                return new_chromosomes

            else:
                # Otherwise add it in
                new_chromosomes[len(new_chromosomes.keys())] = new_chrom

    # return the new chromosomes
    return new_chromosomes

def custom_mutate(chromosomes, custom_vals, percent=0.1):
    '''
    Randomly select and mutate a number of genes in the custom chromosome

    :param chromosomes: A dictionary of custom encoded chromosomes to mutate
    :param custom_vals: A list of tuples defining the ranges of the custom vals. Must be the same
                        length and in the same order as the chromosomes (i.e must define every value in the chromosome)
    :param percent: A percentage indicating the chance that the gene will be mutated

    :return: A dictionary of mutated binary encoded chromosomes
    '''


    # Go along every key
    for i in range(len(chromosomes)):
        # For every value we roll a dice and see if it gets mutated
        for j in range(len(custom_vals)):
            dice = random.uniform(0, 1)
            # If it gets mutated, crate a new value from the parameters in custom_vals
            if dice <= percent:
                if type(custom_vals[j][0]) is int:
                    # Generate a random integer between the values
                    chromosomes[i][j] = random.randrange(custom_vals[j][0], custom_vals[j][1])

                # If it is a float
                elif type(custom_vals[j][0]) is float:
                    # Generate a random float between the values
                    chromosomes[i][j] = random.uniform(custom_vals[j][0], custom_vals[j][1])

    # Return the mutated dictionary (Maybe not strictly necessary as the dictionary has already been changed)
    return chromosomes
