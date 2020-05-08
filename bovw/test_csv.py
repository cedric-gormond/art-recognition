from define_class import listArtistCSV, defineClassWithArtists, defineClassWithOnlyFilename
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from accuracy import *
from load_images import *

print("LOADING TRAIN IMAGE ")
#train_images = load_images_from_folder("bovw/dataset/train4")  # take all images category by category 
print("Done \n")

#print(train_images)
#artist = defineClassWithOnlyFilename(train_images)
#print(artist)

test_class = ["A", "D", "C", "B"]
pred_class = ["A", "B", "B", "B"]

labels = ["A", "B", "C", "D"]

cm = confusion_matrix(test_class, pred_class,labels=labels)
print(cm)
plot_confusion_matrix(cm,labels,'Confusion Matrix - Art Recognition ( visuals words)')
