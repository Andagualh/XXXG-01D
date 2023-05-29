import airetro
import hyperparameters
from test import test
import train

select = 0
while(select == 0):
    print("Select which script you want to run: \n1. Hyperparameter Optimization \n2. Test a Neural Network \n3. Train a new neural network \nType \'quit\' to cancel the execution.")
    print("Input: ")
    select = input()
    if(select == '1'):
        hyperparameters()
        break
    elif(select == '2'):
        testclass = test()
        testclass.start()
        break
    elif(select == '3'):
        train()
        break
    elif(select == 'quit'):
        break
    else:
        print("Please select one of the available options")
        select = 0


