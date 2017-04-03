"""Module used to read data-files or write results.

expected file hierarchy :
    /main : here lies our code files
        /data
            /Sample1
            /Sample2
            ...
        /res
            /Sample1
            ...
            /replicate_PII_refs::i  with i being an integer designed the number
                                    of references used to replicate PII

"""
import csv
import string
import time
import os


data_sets = ['data/HDI/', 'data/EPI/', 'data/GEQ/', 'data/SHA/']


def open_raw(filename):
    """Open raw csv-data-files.

    All data must be written is csv format.
    Different sections are separated with lines of '#'.
    The structure of the different sections is defined by the first line.
    """
    data = []
    with open(filename, newline='') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in content:
            data.append(row)

    header = data[0]
    weights = []
    coefficients = []
    alternatives = []
    titles = ['alternatives', 'weights', 'coefficients']
    matrix = [alternatives, weights, coefficients]
    i = 0
    ind = titles.index(header[i].lower())
    for row in data[2:]:
        if row[0][0] == '#':
            i += 1
            ind = titles.index(header[i].lower())
        else:
            matrix[ind].append(list(map(lambda x: float(x), row[:])))
    return matrix


def write_raw(alternatives, filename):
    """write classic csv-data-files.

    evaluations ...
    """
    output = open(filename, 'w')
    for alt in alternatives:
        for i in range(len(alt)):
            if i != len(alt)-1:
                print(str(alt[i]), end=',', file=output)
            else:
                print(str(alt[i]), file=output)



