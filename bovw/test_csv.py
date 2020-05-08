from define_class import listArtistCSV, defineClassWithArtists
from load_images import *

print("LOADING TRAIN IMAGE ")
train_images = load_images_from_folder("bovw/dataset/train3")  # take all images category by category 
print("Done \n")

#print(train_images)
artist = defineClassWithArtists(train_images)
print(artist)
