import torch
import torch.nn.functional as F
import sklearn.model_selection


def nn_evaluate(chromosomes, data, labels, folds=5):

    # Dictionary of output accuracies
    output_accuracy = {}
    output_loss = {}

    # Now we go through each chromosome, train a nn using the parameters and evaluate the nn
    for key in chromosomes:

        # Get the nn parameters from the chromosome
        n_h1 = chromosomes[key][0]
        n_h2 = chromosomes[key][1]
        learning_rate = chromosomes[key][2]
        num_epochs = chromosomes[key][3]

        # Container for chromosome accuracy and losee
        chromosome_accuracy = []
        chromosome_loss = []

        # kfold splitter
        kfold = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)

        # Go through each fold
        for train_index, test_index in kfold.split(data, labels):

            # Change dataframe to array (have to create a new variable for some reason)
            data_array = data.to_numpy()

            # Define tensors here to avoid a mysterious bug
            x = torch.tensor(data_array, dtype=torch.float)
            y = torch.tensor(labels, dtype=torch.long)

            # Split the tensors based upon the fold's indices
            x_train = x[train_index]
            x_test = x[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            # initialize the network
            n_input = data.shape[1]
            net = torch.nn.Sequential(
                        torch.nn.Flatten(),
                        torch.nn.Linear(n_input,n_h1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_h1, n_h2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_h2, 2)
                    )

            loss_func = torch.nn.CrossEntropyLoss()
            optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

            # Train the network
            for epoch in range(0, num_epochs):

                # Forward pass
                y_pred = net(x_train)
                # Compute loss
                loss = loss_func(y_pred, y_train)
                # Clear the gradients before running the backward pass.
                net.zero_grad()
                # Perform backward pass
                loss.backward()
                # update the optimiser paramaters
                optimiser.step()

            # Evaluate this folds accuracy
            y_pred_test = net(x_test)
            _, predicted = torch.max(F.softmax(y_pred_test, 1), 1)

            # calculate accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == y_test.data.numpy()

            # Get the loss
            loss = loss_func(y_pred, y_train)

            # Accuracy and loss for this fold of this chromosome
            chromosome_accuracy.append(sum(correct)/total)
            chromosome_loss.append(float(loss))

        # Calculate average fold accuracy and average fold loss
        output_accuracy[key] = (sum(chromosome_accuracy)/len(chromosome_accuracy))
        output_loss[key] = sum(chromosome_loss)/len(chromosome_loss)

    # Return the final accuracies and losses
    return output_accuracy, output_loss


def nn_baseline(data, labels, n_h1=10, n_h2=10, num_epochs=150, learning_rate = 0.1, folds=5):
    # A naive neural network for getting a baseline accuracy for comparison reasons
    fold_acc = []
    fold_loss = []

    # kfold splitter
    kfold = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)

    # Go through each fold
    for train_index, test_index in kfold.split(data, labels):

        # Change dataframe to array (have to create a new variable for some reason)
        data_array = data.to_numpy()

        # Define tensors here to avoid a mysterious bug
        x = torch.tensor(data_array, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Split the tensors based upon the fold's indices
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # initialize the network
        n_input = data.shape[1]
        net = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(n_input,n_h1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_h1, n_h2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_h2, 2)
                    )

        loss_func = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

        # Train the network
        for epoch in range(0, num_epochs):

            # Forward pass
            y_pred = net(x_train)
            # Compute loss
            loss = loss_func(y_pred, y_train)
            # Clear the gradients before running the backward pass.
            net.zero_grad()
            # Perform backward pass
            loss.backward()
            # update the optimiser paramaters
            optimiser.step()

        # Evaluate this folds accuracy
        y_pred_test = net(x_test)
        _, predicted = torch.max(F.softmax(y_pred_test, 1), 1)

        # calculate accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == y_test.data.numpy()

        # Get the loss
        loss = loss_func(y_pred, y_train)

        # Accuracy and loss for this fold of this chromosome
        fold_acc.append(sum(correct)/total)
        fold_loss.append(float(loss))

    # Calculate average fold accuracy and average fold loss
    output_accuracy = (sum(fold_acc)/len(fold_acc))
    output_loss = sum(fold_loss)/len(fold_loss)

    # Return the final accuracies and losses
    return output_accuracy, output_loss