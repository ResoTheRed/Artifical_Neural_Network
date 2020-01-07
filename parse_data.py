import random as r
import os

# delete all created files
def clean():
    # if os.path.exists("X.txt"):
    #     os.remove("X.txt")
    # if os.path.exists("Y.txt"):
    #     os.remove("Y.txt")
    if os.path.exists("test_X.txt"):
        os.remove("test_X.txt")
    if os.path.exists("train_X.txt"):
        os.remove("train_X.txt")
    if os.path.exists("test_Y.txt"):
        os.remove("test_Y.txt")
    if os.path.exists("train_Y.txt"):
        os.remove("train_Y.txt")
    # if os.path.exists("mixed_data.txt"):
    #     os.remove("mixed_data.txt")

def mix_up_data():
    indexes = []
    data = {}
    count = 1
    for i in range(150):
        indexes.append(i+1)
    r.shuffle(indexes)
    print(indexes)
    fr = open("iris.data.txt","r")
    fw = open("mixed_data.txt","a")
    line = fr.readline()
    while line:
        data[str(count)] = line
        count +=1
        line = fr.readline()
    for i in range(len(data)):
        fw.write(data[str(indexes[i])])



# parse original dataset
def parse_data():
    fr = open("mixed_data.txt","r")
    x = open("X.txt","a+")
    y = open("Y.txt","a+")
    line = fr.readline()

    while line:
        arr = line.split(",")
        temp = ""
        for i in range(len(arr)):
            if i == len(arr)-1:
                temp+="\n"
                x.write(temp)
                y.write(convert_Y_to_int(arr[i]))
            else:
                temp +=str(arr[i])+" "
        line = fr.readline()
    x.close()
    y.close()
    fr.close()

def convert_Y_to_int(line_y):
    temp = ""
    if line_y=="Iris-setosa\n":
        temp = "1 0 0\n"
    if line_y=="Iris-versicolor\n":
        temp = "0 1 0\n"
    if line_y=="Iris-virginica\n":
        temp = "0 0 1\n"
    return temp

#get test and training data from 1 to 10 for FVC
def fcv(target):
    x = open("X.txt","r")
    y = open("Y.txt","r")
    test_x = open("test_X.txt","a+")
    test_y = open("test_Y.txt","a+")
    train_x = open("train_X.txt","a+")
    train_y = open("train_Y.txt","a+")

    line_x = x.readline()
    line_y = y.readline()
    count = 1
    while line_x:
        if count == target:
            test_x.write(line_x)
            test_y.write(line_y)
        else:
            train_x.write(line_x)
            train_y.write(line_y)
        count +=1
        if count == 11:
            count = 1
        line_x = x.readline()
        line_y = y.readline()
    x.close()
    y.close()
    test_x.close()
    test_y.close()
    train_x.close()
    train_y.close()

def run(target):
    clean()
    mix_up_data()
    parse_data()
    fcv(target)

clean()
fcv(9)



