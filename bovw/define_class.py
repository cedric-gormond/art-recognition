import os 
import csv
import sys
import re

def defineClass(images):
    class_vector = []    
    for image in images:
        class_vector.append(int(os.path.splitext(image)[0]))

    return class_vector

def defineClassWithArtists(images, csv_filename='bovw/dataset/csv/small_data.csv'):
    class_vector = []

    for image in images:
        with open(csv_filename) as cvs_file:
            reader = csv.reader(cvs_file, delimiter=',')
            for row in reader:
                if os.path.splitext(image)[0] == row[0]:   
                    class_vector.append(row[1])

    return class_vector

def defineClassWithOnlyFilename(images):
    class_vector = []

    for image in images:
        if(re.search("^Pablo_Picasso_\d*", os.path.splitext(image)[0])):
            class_vector.append("Pablo Picasso")
        elif(re.search("^Vincent_van_Gogh_\d*", os.path.splitext(image)[0])):
            class_vector.append("Vincent van Gogh")
        elif(re.search("^Claude_Monet_\d*", os.path.splitext(image)[0])):
            class_vector.append("Claude Monet")
        elif(re.search("^Frida_Kahlo_\d*", os.path.splitext(image)[0])):
            class_vector.append("Frida Kahlo")
        elif(re.search("^Gustav_Klimt_\d*", os.path.splitext(image)[0])):
            class_vector.append("Gustav Klimt")
        elif(re.search("^Rene_Magritte_\d*", os.path.splitext(image)[0])):
            class_vector.append("Rene Magritte")
        elif(re.search("^Andy_Warhol_\d*", os.path.splitext(image)[0])):
            class_vector.append("Andy Warhol")
        elif(re.search("^Henri_Matisse_\d*", os.path.splitext(image)[0])):
            class_vector.append("Henri Matisse")
        elif(re.search("^Paul_Cezanne_\d*", os.path.splitext(image)[0])):
            class_vector.append("Paul Cezanne")
        elif(re.search("^Gustave_Courbet_\d*", os.path.splitext(image)[0])):
            class_vector.append("Gustave Courbet")
        elif(re.search("^Edouard_Manet_\d*", os.path.splitext(image)[0])):
            class_vector.append("Edouard Manet")
        elif(re.search("^Joan_Miro_\d*", os.path.splitext(image)[0])):
            class_vector.append("Joan Miro")
        elif(re.search("^Salvador_Dali_\d*", os.path.splitext(image)[0])):
            class_vector.append("Salvador Dali")
        elif(re.search("^Piet_Mondrian_\d*", os.path.splitext(image)[0])):
            class_vector.append("Piet Mondrian")
        elif(re.search("^Pierre-Auguste_Renoir_\d*", os.path.splitext(image)[0])):
            class_vector.append("Pierre-Auguste Renoir")

    return class_vector

def matchArtist(id,csv_filename='bovw/dataset/csv/all_data_info.csv'):
    # cr=csv.reader(open("all_data_info.csv","r",encoding="utf8"))
    cr=csv.reader(open(csv_filename,"r",encoding="utf8"))
    next(cr)
    Artist_and_id=[]
    for row in cr:
        id=int(row[-1][:-4])
        Artist_and_id.append([id,row[0]])

    Artist_and_id.sort()


def listArtistCSV(csv_filename='bovw/dataset/csv/all_data_info.csv'):
    #    cr=csv.reader(open("all_data_info.csv","r",encoding="utf8"))
    cr=csv.reader(open(csv_filename,"r",encoding="utf8"))
    next(cr)
    Artist_and_id=[]
    for row in cr:
        id=int(row[-1][:-4])
        Artist_and_id.append([id,row[0]])

    return Artist_and_id.sort()