Put here network.pt of model that you want to train with floydHub. If this
directory is empty, a new model will be created.

To train with floyd:
    Go to main project directory.
    You need to have floyd account and be logged in, and you have to initialize
    floyd repository in main project directory.
    Then, execute following command:

    floyd run --cpu --env pytorch-0.2 'python3 src/train_floyd.py'

    After a while, go to the project page on floydhub and gather output model.


