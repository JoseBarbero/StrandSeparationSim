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
        partition (str): train, test or val
        categories (str) : A subset of ["OPN", "BUB10", "BUB12", "BUB8", "VRNORM"]

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
        
        # Tengo muchas dudas de que esto esté bien.
        # En teoría habría que coger las 28 temperaturas distintas de 1 misma instancia.
        # Eso implicaría que la Y sería un vector de 1s y 0s en función de si se abre a esa temperatura.
        #
        # No me queda claro qué es lo que estoy haciendo ahora. Diría que se están cogiendo los datos de 28 instancias positivas a diferentes temperaturas y se le pone 1 sola etiqueta.
        # No tiene por qué estar mal pero no sé si tiene tanto sentido como lo otro.
        # 
        # Es que las propias secuencias originales ya están etiquetadas como positivas y negativas. No entiendo.

        instances_pos = np.asarray([instances for temp, instances in sorted(data_pos)])
        instances_neg = np.asarray([instances for temp, instances in sorted(data_neg)])


        for row in range(instances_pos.shape[1]):
            X.append(instances_pos[:,row,:])
            y.append(1)

        for row in range(instances_neg.shape[1]):
            X.append(instances_neg[:,row,:])
            y.append(0)

    return np.asarray(X), np.asarray(y)
    

def seq_to_array(seqfile):
    X_fw = []
    X_rv = []
    basetovalue_fw = {'A': 0.25, 'T': 0.5, 'G': 0.75, 'C': 1}
    basetovalue_rv = {'A': basetovalue_fw['T'], 'T': basetovalue_fw['A'], 'G': basetovalue_fw['C'], 'C': basetovalue_fw['G']}
    with open(seqfile, "r") as _file:
        for line in _file:
            line_values_fw = []
            line_values_rv = []
            for base in line[:-1]:
                line_values_fw.append(basetovalue_fw.get(base.upper(), 0))     # Aquí no estoy diferenciando entre caps y no, por lo tanto no diferencio entre secuencia 3' y 5'.
                line_values_rv.append(basetovalue_rv.get(base.upper(), 0))
            X_fw.append(line_values_fw)
            X_rv.append(line_values_rv)
    return np.asarray(X_fw), np.asarray(X_rv)


def seq_to_onehot_array(seqfile):
    X_fw = []
    X_rv = []
    basetovalue_fw = {'A': np.array([1, 0, 0, 0]), 'T': np.array([0, 1, 0, 0]), 'G': np.array([0, 0, 1, 0]), 'C': np.array([0, 0, 0, 1])}
    basetovalue_rv = {'A': basetovalue_fw['T'], 'T': basetovalue_fw['A'], 'G': basetovalue_fw['C'], 'C': basetovalue_fw['G']}
    with open(seqfile, "r") as _file:
        for line in _file:
            line_values_fw = []
            line_values_rv = []
            for base in line[:-1]:
                line_values_fw.append(basetovalue_fw.get(base.upper(), [0, 0, 0, 0]))     # Aquí no estoy diferenciando entre caps y no, por lo tanto no diferencio entre secuencia 3' y 5'.
                line_values_rv.append(basetovalue_rv.get(base.upper(), [0, 0, 0, 0]))
            X_fw.append(line_values_fw)
            X_rv.append(line_values_rv)
    return np.asarray(X_fw), np.asarray(X_rv)


def read_data_st_withseq(directory, partition, categories):
    """
    Reads csv data to numpy stacking temperatures. With the original sequence in a 2nd channel and the reversed sequence in the 3rd channel.

    Args:
        directory (str): Directory containing the files to read.
        partition (str): train, test or val.
        categories (str) : A subset of ["OPN", "BUB10", "BUB12", "BUB8", "VRNORM"]

    Returns:
        tuple: (np.array, np.array): X, Y data.
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
                genome = 'hg16' if partition in ['train', 'val'] else 'hg17'

                seqfile = directory+'/onlyseq.TSS'+tag+'FineGrained.'+genome+'-'+partition+'.'+tag          # TODO Don't hardcode this
                
                seq_data_fw = seq_to_array(seqfile)[0]
                seq_data_rv = seq_to_array(seqfile)[1]

                prob_data = file_to_array(directory+"/"+filename)[0]

                both_data = [prob_data, seq_data_fw, seq_data_rv]

                if tag == "pos":
                    data_pos.append((temp, both_data))
                elif tag == "neg":
                    data_neg.append((temp, both_data))

        instances_pos = np.asarray([instances for temp, instances in sorted(data_pos)])
        instances_neg = np.asarray([instances for temp, instances in sorted(data_neg)])
        instances_pos = np.swapaxes(instances_pos, 0, 2)
        instances_neg = np.swapaxes(instances_neg, 0, 2)
        instances_pos = np.moveaxis(instances_pos, 1, -1)
        instances_neg = np.moveaxis(instances_neg, 1, -1)

        # TODO This is pretty slow
        for row in instances_pos:
            X.append(row)
            y.append(1)

        for row in instances_neg:
            X.append(row)
            y.append(0)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def read_data_st_withseq_onehot(directory, partition, categories):
    """
    Reads csv data to numpy stacking temperatures. With the original sequence in a 2nd channel and the reversed sequence in the 3rd channel.

    Args:
        directory (str): Directory containing the files to read.
        partition (str): train, test or val.
        categories (str) : A subset of ["OPN", "BUB10", "BUB12", "BUB8", "VRNORM"]

    Returns:
        tuple: (np.array, np.array): X, Y data.
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
                genome = 'hg16' if partition in ['train', 'val'] else 'hg17'

                seqfile = directory+'/onlyseq.TSS'+tag+'FineGrained.'+genome+'-'+partition+'.'+tag          # TODO Don't hardcode this
                
                seq_data_fw, seq_data_rv = seq_to_onehot_array(seqfile)

                prob_data = file_to_array(directory+"/"+filename)[0]
                both_data = np.asarray([prob_data,
                                        seq_data_fw[:, :, 0], seq_data_fw[:, :, 1], seq_data_fw[:, :, 2], seq_data_fw[:, :, 3],
                                        seq_data_rv[:, :, 0], seq_data_rv[:, :, 1], seq_data_rv[:, :, 2], seq_data_rv[:, :, 3]])

                if tag == "pos":
                    data_pos.append((temp, both_data))
                elif tag == "neg":
                    data_neg.append((temp, both_data))

        instances_pos = np.asarray([instances for temp, instances in sorted(data_pos)])
        instances_neg = np.asarray([instances for temp, instances in sorted(data_neg)])
        instances_pos = np.swapaxes(instances_pos, 0, 2)
        instances_neg = np.swapaxes(instances_neg, 0, 2)
        instances_pos = np.moveaxis(instances_pos, 1, -1)
        instances_neg = np.moveaxis(instances_neg, 1, -1)

        # TODO This is pretty slow
        for row in instances_pos:
            X.append(row)
            y.append(1)

        for row in instances_neg:
            X.append(row)
            y.append(0)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y