import numpy as np
import pandas as pd
import os


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


def read_from_directory(directory):
    """
    Gets the information on every file in a certain directory.

    Returns:
        data: List containing tuples (data, class) of every file in that directory.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.startswith("OPN"):
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
    


def read_data(directory):

    # Read the data    
    data = read_from_directory(directory)
    
    # Create instances with equal dimensions
    X = []
    Y = []
    chunk_size = 200 # Pongo 200 para hacer los ejemplos cuadrados porque son 200 columnas
    
    for instance, tag in data:
        splitted_data = split_instance(instance, chunk_size)
        X.extend(splitted_data)
        Y.extend([tag] * len(splitted_data))
    
    return (X, Y)