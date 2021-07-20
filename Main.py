# Main python file
import ea_funcs
import nn_funcs
import pandas as pd



# This file calls the functions from ea_funcs.py, nn_funcs.py and executes preprocessing.py

# ************** Preprocessing ***************

# If this is the first time running this file, execute the following
# command (or run the preprocessing.py file manually). preprocessing.py must
# be in the working directory.

# preprocessing.py processes the raw data by:
#   1 - Removing unwanted columns
#   2 - Removing columns without any data
#   3 - Splitting labels into a separate file
#   4 - putting all data on a standardised scale
#   5 - replacing significant outliers with the mean value
#   6 - removing some columns to make the solution a bit more difficult
#   7 - saves the edited data to the working directory

#exec(open('preprocessing.py').read())


# ************** Read in files *************
# read in data and the labels
df = pd.read_csv('features_cleaned_scaled.csv')
labels = pd.read_csv('labels.csv')

# Convert labels to a flat list for the svm classifier
labels = labels['label'].values.tolist()

# ************** Global parameters *************
#
# How many generations to run
search_gens = 100
optim_gens = 30
# How many chromosomes to start with
search_chroms = 150
optim_chroms = 100 # Will crash if there aren't enough chroms to reproduce (should be above 30)
# What percentage of chromosomes should be allowed to 'reproduce'
search_cutoff = 0.45
optim_cutoff = 0.45
# Number of new chromosomes made in the crossover process
search_spawn = search_chroms
optim_spawn = optim_chroms
# What percent of bits in the chromosome should mutate (the exact number is randomly selected from between
# this range for the svm
search_mutate = (0.01, 0.05)
optim_mutate = 0.30

# ******************************************** SEARCH ***************************************************


# How many features will be "on" for those chromosomes (how many 1's in the chromosome)
starting_features = 1
# Max number of features that a chromosome can have enabled (max number of 1's in the chromosome)
max_features = 5
# Number of folds in k-fold cross validation (min is 2)
n_folds = 5

# Containers to track results
highest_accuracy = 0
best_features = []
svm_generation = []
svm_accuracy_list = []
unique_features = []

# Seed for reproducability
seed_n = 1001001001

# Create n chromosomes with the desired number of starting features
chromosomes = ea_funcs.create_binary_chroms(search_chroms, len(df.columns), starting_features)


for i in range(0, search_gens):
    print("Generation {}:".format(i+1))
    svm_generation.append(i+1)

    # Keep track of the total number of unique features in the system (i.e the diversity)
    all_features = set()
    for key in chromosomes:
        get_features = [i for i, val in enumerate(chromosomes[key]) if val !=0]
        all_features.update(get_features)
    # Add the number in the container
    unique_features.append(len(all_features))

    # Evaluate the chromosomes
    chromosomes_evaluated = ea_funcs.evaluate_chroms_svm(chromosomes, data=df,
                                                         labels=labels,
                                                         folds=n_folds)

    # Find the highest accuracy acheived
    accuracy = max(chromosomes_evaluated.values())
    svm_accuracy_list.append(accuracy)
    # Find which chromosome got the highest accuracy
    max_key = max(chromosomes_evaluated,
                  key=chromosomes_evaluated.get)

    # If a new maximum accuracy has been found, record it and print the accuracy and features
    if accuracy > highest_accuracy:
        highest_accuracy=accuracy
        best_features = [i for i, val in enumerate(chromosomes[max_key]) if val != 0]
        print("New highest accuracy of {} using features {}".format(highest_accuracy, best_features))

    # If we have hit max accuracy
    if accuracy == 1:
        features = df.columns[best_features]
        print("\n\nAchieved 100% accuracy on generation {} using the features {}".format(i+1, features))
        break

    # Select the best x% of chromosomes for crossover
    survivors = ea_funcs.select(chromosomes, chromosomes_evaluated, percentage=search_cutoff)

    # Perform two point crossover on the survivors to create n new chromosomes
    new_chroms = ea_funcs.binary_crossover(survivors, n_chroms_out=search_spawn)

    # Mutate an amount of chromosomes randomly
    chromosomes = ea_funcs.mutate(new_chroms)

    # Randomly turn off features in the chromosomes until under a cutoff
    chromosomes = ea_funcs.trim(chromosomes, max_features=max_features)

    # If we're on the final generation, print the best results
    if i == search_gens-1:
        features = df.columns[best_features]
        print("\n\n\nFinished! Highest accuracy of {} using features {}".format(highest_accuracy, features))


# *************************************** Optimize **************************************************


# Now that we have selected some of the best features from the data set, we can use
# use those features for classification with a neural network. We will use an EA to optimize
# the network.

# The EA will optimize four settings: the number of neurons in the two hidden layers, the learning rate and
# number of epochs to train on. The list of tuples below defines the boundaries for each parameter
custom_vals = [(1, 40), (1, 40), (0.001, 0.3), (5, 150)]

# Optimise based on loss or accuracy
optimise_on = "loss" # or accuracy

# Subset the best data
#best_features = [78, 108, 121, 127], these are the features referenced in the assignment

best_data = df.iloc[:, best_features]

# Containers to track results
highest_accuracy = 0
lowest_loss = 1
nn_generation = []
nn_accuracy_list = []
loss_list = []
settings_list = []


# Create n chromosomes with the desired number of starting features
chromosomes = ea_funcs.create_custom_chroms(optim_chroms, custom_vals)

# Run the optimisation
for i in range(optim_gens):
    print("Generation {}:".format(i+1))
    nn_generation.append(i+1)

    # Evaluate the chromosomes
    chrom_accuracies, chrom_losses = nn_funcs.nn_evaluate(chromosomes, data=best_data,
                                                         labels=labels,
                                                         folds=n_folds)

    # Record accuracy, loss and settings for highest accuracy chromosome in the generation
    if optimise_on == "accuracy":

        # Find the highest accuracy achieved in this generation
        accuracy = max(chrom_accuracies.values())

        # Find which chromosome got the highest accuracy
        max_key = max(chrom_accuracies,
                      key=chrom_accuracies.get)

        # Find the best settings for this generation
        acc_settings = chromosomes[max_key]

        # Record these values in containers
        nn_accuracy_list.append(accuracy)
        loss_list.append(chrom_losses[max_key])
        settings_list.append(acc_settings)

        if accuracy > highest_accuracy:
            highest_accuracy=accuracy
            best_acc_settings=acc_settings
            print("****** New highest accuracy of {} with loss {} using settings {} ********".format(
                highest_accuracy, chrom_losses[max_key], best_acc_settings))

        else:
            print("accuracy of {} with loss {} using the settings {}".format(accuracy, chrom_losses[max_key],
                                                                             acc_settings))

    # If a new minimum loss has been found, print the loss and settings
    if optimise_on == 'loss':

        # Find the lowest loss
        loss = min(chrom_losses.values())

        # Find which chromosome got the lowest loss
        min_key = min(chrom_losses,
                      key=chrom_losses.get)
        # Find the best settings for this generation
        loss_settings = chromosomes[min_key]

        # Add these values to containers
        loss_list.append(loss)
        nn_accuracy_list.append(chrom_accuracies[min_key])
        settings_list.append(loss_settings)

        if loss < lowest_loss:
            lowest_loss=loss
            best_loss_settings=loss_settings
            print("****** New lowest loss of {} with accuracy {} using settings {} ********".format(
                lowest_loss, chrom_accuracies[min_key], best_loss_settings))

        else:
            print("loss of {} with accuracy {} using the settings {}".format(loss,
                                                                             chrom_accuracies[min_key], loss_settings))

    # Select the best x% of chromosomes for crossover using accuracy or loss
    if optimise_on == "accuracy":
        survivors = ea_funcs.select(chromosomes, chrom_accuracies, percentage=optim_cutoff)
    elif optimise_on == "loss":
        # Subtract the losses from 1 so the highest loss is now the best chromosome
        for key in chrom_losses:
            chrom_losses[key] = 1 - chrom_losses[key]
        # Now we can just optimise on loss as before with accuracy
        survivors = ea_funcs.select(chromosomes, chrom_losses, percentage=optim_cutoff)
    else:
        print("invalid optimzer, use accuracy or loss")
        break

    # Perform custom crossover (which is simply the average value)
    new_chroms = ea_funcs.custom_crossover(survivors, n_chroms_out=optim_spawn)

    # Mutate an amount of chromosomes randomly
    chromosomes = ea_funcs.custom_mutate(new_chroms, custom_vals, optim_mutate)

    # If we're on the final generation, print the best results
    if i == optim_gens-1:
        if optimise_on == 'accuracy':
            print("\n\n\nFinished! Highest accuracy of {} using settings {}".format(highest_accuracy, best_acc_settings))
        else:
            print("\n\n\nFINISHED! Lowest loss of {} using settings {}".format(lowest_loss, best_loss_settings))

# Run the baseline sim for comparison
df_base_acc, df_base_loss = nn_funcs.nn_baseline(df, labels)
trimmed_base_acc, trim_base_loss = nn_funcs.nn_baseline(best_data, labels)

print("Accuracy for the entire dataset using a naive NN was {} with loss {}".format(df_base_acc, df_base_loss))
print("Trimmed accuracy for the dataset using a naive NN was {} with loss {}".format(trimmed_base_acc, trim_base_loss))
