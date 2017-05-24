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
    thresholds = []
    titles = ['alternatives', 'weights', 'coefficients', 'thresholds']
    matrix = [alternatives, weights, coefficients, thresholds]
    i = 0
    ind = titles.index(header[i].lower().strip())
    for row in data[2:]:
        if row[0][0] == '#':
            i += 1
            ind = titles.index(header[i].lower().strip())
        else:
            matrix[ind].append(list(map(lambda x: float(x), row[:])))
    for i in range(1, len(matrix)):
        if matrix[i]:
            matrix[i] = matrix[i][0]

    return matrix


def open_raw_RS(filename):
    """Open file with RS separated by some ###."""
    data = []
    with open(filename, newline='') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in content:
            data.append(row)

    all_RS = []
    RS = []
    for row in data:
        if row[0][0] == '#':
            all_RS.append(RS)
            RS = []
        else:
            RS.append(list(map(lambda x: float(x), row[:])))
    return all_RS

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



