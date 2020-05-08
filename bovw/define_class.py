import os 
import csv
import sys

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

def matchArtist(id,csv_filename='bovw/dataset/csv/all_data_info.csv'):
    #    cr=csv.reader(open("all_data_info.csv","r",encoding="utf8"))
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