import torch 
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
from sklearn import metrics

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

import torchvision.transforms as transforms
import pandas as pd

import copy


class MNIST_data:
    """
    A class for handling the creation of the train, validation and test dataset, pytorch ready.

    train_path : string
        path for the train images
    test_path : string
        path for the test images
    batch_size : int
    train_split_ration : float [0, 1]
        ratio of training images in the train image folder.
        The rest of the images will be allocated to the valdiation dataset
    """
    def __init__(self, train_path, test_path, batch_size, train_split_ratio=0.8):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_split_ratio = train_split_ratio

    class MNIST_train(Dataset):
        def __init__(self, pixels, labels):
            self.pixels = torch.tensor(pixels, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0  # Normalize to [0, 1]
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.pixels[idx], self.labels[idx]

    class MNIST_test(Dataset):
        def __init__(self, pixels, test_df):
            self.pixels = torch.tensor(pixels, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0  # Normalize to [0, 1]
            self.test_df = test_df

        def __len__(self):
            return len(self.test_df)

        def __getitem__(self, idx):
            return self.pixels[idx]


    def make_train(self, in_batch_size=None):
        """
        Initializes train and validation datasets.

        RETURN : Dataloader
            train_loader
            val_loader
        """
        batch_size = in_batch_size or self.batch_size
        train_df = pd.read_csv(self.train_path)
        labels = train_df['label'].values
        pixels = train_df.drop(columns=['label']).values
        mnist_dataset = self.MNIST_train(pixels, labels)

        train_size = int(self.train_split_ratio * len(mnist_dataset))
        val_size = len(mnist_dataset) - train_size
        train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader
    
    def make_test(self, in_batch_size=None):
        """
        Initializes a test dataset to later get a Kaggle score for the competition.

        RETURN : Dataloader
            test_laoder
        """
        batch_size = in_batch_size or self.batch_size # should not be necessary as test datas are not used in the grid_search
        test_df = pd.read_csv(self.test_path)
        pixels = test_df.values

        mnist_dataset_test = self.MNIST_test(pixels, test_df)
        test_loader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)
        return test_loader
    
    def make_submission_train(self, in_batch_size=None):
        """
        Initializes train dataset with the whole dataset (to train the fine-tuned model on
            a maximum amount of data).

        RETURN : Dataloader
            train_loader
        """
        batch_size = in_batch_size or self.batch_size
        train_df = pd.read_csv(self.train_path)
        labels = train_df['label'].values
        pixels = train_df.drop(columns=['label']).values
        dataset = self.MNIST_train(pixels, labels)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=False)


        return train_loader


class MNIST:
    """
    A class for handling the training, evaluation and submission of the MNIST kaggle challenge.

    mnist_data : MNIST_data
        An instance of MNIST_data containing both train, validation and test dataset (pytorch ready)
    """
    def __init__(self, mnist_data):
        self.mnist_data = mnist_data

        # loaders :
        self.train_loader, self.val_loader = mnist_data.make_train()
        self.test_loader = mnist_data.make_test()

    def train(self, model, criterion, optimizer, epochs=10, verbose = 0):
        """
        Basic PyTorch model training algorithm.

        model     : torch.nn.Module
        criterion : torch.nn.Module
        optimizer : torch.optim.Optimizer
        epochs    : int
        verbose   : int, optional
            velue 0 recommended when performing a GridSearch.

        RETURN : dict
            dictionary containing the training loss for each batch.
        """

        output = {'training_loss': []}  
        for epoch in range(epochs):
            if verbose : print(str(epoch) + " / " + str(epochs))
            for i, (image, pred) in enumerate(self.train_loader):
                optimizer.zero_grad()
                z = model(image)
                loss = criterion(z, pred)
                loss.backward()
                optimizer.step()
                output['training_loss'].append(loss.data.item())
        return output
    
    def train_to_submit(self, model, criterion, optimizer, epochs=10, verbose = 0):
        """
        Same as train but with the whole training dataset to optimize training.
        Evaluating a model -on val_loader- trained with this function would be conceptualy wrong.
        """
        self.train_loader = self.mnist_data.make_submission_train()
        output = {'training_loss': []}  
        for epoch in range(epochs):
            if verbose : print(str(epoch) + " / " + str(epochs))
            for i, (image, pred) in enumerate(self.train_loader):
                optimizer.zero_grad()
                z = model(image)
                loss = criterion(z, pred)
                loss.backward()
                optimizer.step()
                output['training_loss'].append(loss.data.item())
        return output
    

    def evaluation(self, model):
        """
        Evaluate the model's accuracy on the validation dataset.

        model : torch.nn.Module

        RETURN : float
            accuracy score on the validation set.
        """
        model.eval()
        count = 0
        for img, label in self.val_loader:
            for i in range(len(label)):
                if model(img[i]).argmax() == label[i] :
                    count = count+1
        return count/(len(self.val_loader)*len(label))
    
    def confusion_matrix_evaluation(self, model):
        """
        Dispalys the confusin matrix for evaluated data over the validation dataset.

        model : torch.nn.Module

        RETURN : X
            X
        """
        model.eval()
        y_pred = []
        y_true = []
        for img, label in self.val_loader:
            for i in range(len(label)):
                y_pred.append(model(img[i]).argmax())
                y_true.append(label[i])

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cm_display.plot()
        plt.show()
    

    def submit(self, model):
        """
        Generates predictions for the test set and saves them to a CSV file.
        The file {submission.csv} will be wiped and written on OR create
        (depending on if the file already exists or not).
        
        model : torch.nn.Module
        
        RETURN : None
            Saves the predictions to "submission.csv" formated to kaggle's
            submission standards.
        """
        f = open("submission.csv", "w+")
        f.truncate()
        f.write("ImageId,Label\n")
        i = 1
        for x in self.test_loader:
            batch_pred = model(x)
            for elt in batch_pred:
                f.write(str(str(i) + "," + str(elt.argmax().numpy()) + "\n"))
                i = i + 1
        f.close()
        print("File created, ready to submit.")
    


class MNIST_gridSearch:
    """
        A class to perform a gridsearch (as seen in SciKit) on a pytorch model
        over the following hyperparameters :
            model          : torch.nn.Module 
            mnist          : MNIST
            criterions     : list of torch.nn.Module
            optimizers     : list of torch.optim.Optimizer
            epochs         : list of int
            learning_rates : list of float
            batch_sizes    : list of int
        
        The optimizers and criterions need to be able to work without changing the model's dimension
        or it will raise an error and stop the program
    """
    def __init__(self, model, mnist: MNIST, criterions, optimizers,
                 epochs = [10],
                 learning_rates = [0.001],
                 batch_sizes = [32]):
        """
        Initializes the grid search configuration.
        
        total_iterations : int
            amount of possible combiantions for the input parameters.
        """
        self.model = model
        self.mnist = mnist
        self.criterions = criterions
        self.optimizers = optimizers
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.total_iterations = len(criterions) * len(optimizers) * len(learning_rates) * len(batch_sizes) * len(epochs)

    def gridSearch(self, verbose = 0) :
        """
        Performs a GridSearch over specified hyperparameters. Every combination of hyperparameters will be tested,
        this can be a very long process
        
        verbose : int
            set to 1 to display (current_iteration)/(toral_iterations) during the process. (highly recommended)

        RETURN : list
            Outputs the list of the best hyperparameters combination in the provided grid and its score.
            [optimizer, criterion, epoch, learning_rate, batch_size, score]
        
        """
        # max = [0, 0, 0, 0, 0, 0] # opt, crit, epoch, l_rate, batch_size, score
        best_param = {"optimizer": None, "criterion": None, "epochs": None, 
                       "learning_rate": None, "batch_size": None, "score": 0.0}
        iteration = 0
        best_model = None

        for batch_size in self.batch_sizes:
            self.mnist.train_loader, self.mnist.val_loader = self.mnist.mnist_data.make_train(batch_size)
            for optimizer in self.optimizers:
                for criterion in self.criterions:
                    for epoch in self.epochs:
                        for l_rate in self.learning_rates:
                            current_model = type(self.model)() # creating a new instance
                            current_model.train() ##

                            iteration += 1
                            optim = optimizer(current_model.parameters(), lr=l_rate)
                            self.mnist.train(current_model, criterion(), optim, epoch)
                            score = self.mnist.evaluation(current_model)

                            if  verbose == 1 : print(f"Iteration {iteration} / {self.total_iterations} | score : {score}")
                            elif verbose >=2 : print(f"Iteration {iteration} / {self.total_iterations} | score : {score} | epoch : {epoch}")
                            if score > best_param["score"] :
                                best_param.update({"optimizer": optimizer, "criterion": criterion, 
                                                    "epochs": epoch, "learning_rate": l_rate, "batch_size": batch_size, 
                                                    "score": score})
                                best_model = copy.deepcopy(current_model)
        
        return best_param, best_model