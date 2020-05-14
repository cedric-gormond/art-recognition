from load_features import *
from accuracy import *
from load_images import *
import time
import json

accuracies = {"a" : 0.3, "b" : 0.4, "c" : 0.3}
k = 400


    
#file_object = open("bovw/results/ACCURACY/trainTEST.csv", "a")

# Wrinting-in 
result = str(k)+ ", " + str(list(accuracies.values())).strip('[]')+ "\n" 
print(result)
#file_object.write(result)
#file_object.close()

    