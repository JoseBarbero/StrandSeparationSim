import numpy as np
import pandas as pd
import os
import re

def file_to_array(inputfile):
    """
    Gets a tsv file and packs it as a numpy 2D array. It returns that array with its correspondent tag.

    Returns:
        data, tag: tuple containing file information into a 2D numpy array and its class.
    """
    data = np.genfromtxt(inputfile, delimiter="\t")

    tag = None
    if inputfile.endswith("pos"):
        tag = 1
    elif inputfile.endswith("neg"):
        tag = 0

    return (data, tag)


def read_from_directory(directory, filter_str):
    """
    Gets the information on every file in a certain directory.

    filter_str (str): Regular expression to filter files in the directory. 

    Returns:
        data: List containing tuples (data, class) of every file in that directory.
    """

    data = []
    for filename in os.listdir(directory):
        if re.match(filter_str, filename):
            print(filename)
            data.append(file_to_array(directory+"/"+filename))
    return data


def split_instance(instance, size):
    """
    Splits an instance into equal-sized chunks of itself.

    Args:
        instance (numpy array): 2D numpy array containing the data.
        size (int): size of the chunks
    
    Returns:

    """
    c = 0
    new_instances = []
    
    # TODO Con esto estoy perdiendo las últimas filas que no llegan a ser 200 para el último bloque
    while c + size < instance.shape[0]:
        new_instances.append(instance[c:c+size])
        c = c + size

    return new_instances
    


def read_data_as_img(directory, filter_str):
    """
    Reads csv data to numpy as if it was an image.

    Args:
        directory (str): Directory containing the files to read.
        filter_str (str): Regular expression to filter files in the directory.

    Returns:
        tuple: (np.array, np.array): X, Y data
    """

    # Read the data    
    data = read_from_directory(directory, filter_str)
    
    # Create instances with equal dimensions
    X = []
    y = []
    chunk_size = 200 # Pongo 200 para hacer los ejemplos cuadrados porque son 200 columnas
    
    for instance, tag in data:
        splitted_data = split_instance(instance, chunk_size)
        X.extend(splitted_data)
        y.extend([tag] * len(splitted_data))
    
    return (np.asarray(X), np.asarray(y))       # TODO esto no es nada eficiente


def read_data_structured(directory, filter_str):

    X = []
    y = []

    for filename in os.listdir(directory):
        if re.match(filter_str, filename):
            instances = file_to_array(directory+"/"+filename)
            X.extend(instances[0])
            y.extend(np.array([instances[1]]*len(instances[0])))

    return (np.asarray(X), np.asarray(y))       # TODO esto no es nada eficiente


def read_data_st(directory, partition, categories):
    """
    Reads csv data to numpy stacking temperatures.

    Args:
        directory (str): Directory containing the files to read.

    Returns:
        tuple: (np.array, np.array): X, Y data
    """

    X = []
    y = []

    for category in categories:
        data_pos = []
        data_neg = []
        for filename in os.listdir(directory):
            if re.match(f"{category}at.*K.*{partition}.*", filename):
                temp = filename.split("at")[1].split("K")[0]
                tag = filename.split(".")[-1]
                if tag == "pos":
                    data_pos.append((temp, file_to_array(directory+"/"+filename)[0]))
                elif tag == "neg":
                    data_neg.append((temp, file_to_array(directory+"/"+filename)[0]))
        instances_pos = np.asarray([instances for temp, instances in sorted(data_pos)])
        instances_neg = np.asarray([instances for temp, instances in sorted(data_neg)])

        for row in range(instances_pos.shape[1]):
            X.append(instances_pos[:,row,:])
            y.append(1)

        for row in range(instances_neg.shape[1]):
            X.append(instances_neg[:,row,:])
            y.append(0)

    return np.asarray(X), np.asarray(y)
    

