import airetro
from hyperparameters import hyperparameters
from test import test
from train import train

select = 0
while(select == 0):
    print("Select which script you want to run: \n1. Hyperparameter Optimization \n2. Test a Neural Network \n3. Train a new neural network \nType \'quit\' to cancel the execution.")
    print("Input: ")
    select = input()
    if(select == '1'):
        hyperclass = hyperparameters()
        hyperclass.train()
        break
    elif(select == '2'):
        testclass = test()
        testclass.start()
        break
    elif(select == '3'):
        trainclass = train()
        trainclass.__init__()
        break
    elif(select == 'quit'):
        break
    else:
        print("Please select one of the available options")
        select = 0


