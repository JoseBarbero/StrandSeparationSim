import numpy as np
import os
import re
from datetime import datetime

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

        instances_pos = np.asarray([instances for temp, instances in sorted(data_pos)])
        instances_neg = np.asarray([instances for temp, instances in sorted(data_neg)])


        for row in range(instances_pos.shape[1]):
            X.append(instances_pos[:,row,:])
            y.append(1)

        for row in range(instances_neg.shape[1]):
            X.append(instances_neg[:,row,:])
            y.append(0)

    return np.asarray(X), np.asarray(y)
    
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



def read_data_channels_onehot(directory, partition, temperatures, categories=["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]):
    """
    Reads csv data to numpy stacking temperatures.

    Args:
        directory (str): Directory containing the files to read.
        partition (str): train, test or val.
        temperatures (list): List of temperatures to read.
        categories (str) : ["OPN", "BUB10", "BUB12", "BUB8", "VRNORM"]

    Returns:
        tuple: (np.array, np.array): X, Y data.
    
    Returned X extra info:
    Size 28 (temperatures) x 200 (bases) x 13 (channels)
        Channel 0: OPN probabilities.
        Channel 1: BUB8 probabilities.
        Channel 2: BUB10 probabilities.
        Channel 3: BUB12 probabilities.
        Channel 4: VRNORM probabilities.
        Channels 5, 6, 7, 8: Forward sequence one-hot encoded.
        Channels 9, 10, 11, 12: Reversed sequence one-hot encoded. 
    """

    X = []
    y = []

    data_pos = []
    data_neg = []

    for temp in temperatures:
        for tag in ['pos', 'neg']:
            hg = '16' if partition in ['train', 'val'] else '17'
            
            opn_file = directory+'/OPNat'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub8_file = directory+'/BUB8at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub10_file = directory+'/BUB10at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub12_file = directory+'/BUB12at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            vrnorm_file = directory+'/VRNORMat'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            seq_file = directory+'/onlyseq.TSS'+tag+'FineGrained.hg'+hg+'-'+partition+'.'+tag

            opn_data = file_to_array(opn_file)[0]
            bub8_data = file_to_array(bub8_file)[0]
            bub10_data = file_to_array(bub10_file)[0]
            bub12_data = file_to_array(bub12_file)[0]
            vrnorm_data = file_to_array(vrnorm_file)[0]
            seq_data_fw, seq_data_rv = seq_to_onehot_array(seq_file)
            #aa_fw_rf1, aa_fw_rf2, aa_fw_rf3, aa_rv_rf1, aa_rv_rf2, aa_rv_rf3 = seq_to_onehot_aminoacids(seq_file)
            seq_data_fw = seq_data_fw[:,50:-50,:]
            seq_data_rv = seq_data_rv[:,50:-50,:]
           

            combined_data = np.asarray([opn_data,
                                        bub8_data,
                                        bub10_data,
                                        bub12_data,
                                        vrnorm_data,
                                        *[seq_data_fw[:, :, i] for i in range(seq_data_fw.shape[2])],
                                        *[seq_data_rv[:, :, i] for i in range(seq_data_rv.shape[2])]])
                                        #*[aa_fw_rf1[:, :, i] for i in range(aa_fw_rf1.shape[2])],
                                        #*[aa_fw_rf2[:, :, i] for i in range(aa_fw_rf2.shape[2])],
                                        #*[aa_fw_rf3[:, :, i] for i in range(aa_fw_rf3.shape[2])],
                                        #*[aa_rv_rf1[:, :, i] for i in range(aa_rv_rf1.shape[2])],
                                        #*[aa_rv_rf2[:, :, i] for i in range(aa_rv_rf2.shape[2])],
                                        #*[aa_rv_rf3[:, :, i] for i in range(aa_rv_rf3.shape[2])]])
            if tag == "pos":
                data_pos.append((temp, combined_data))
            elif tag == "neg":
                data_neg.append((temp, combined_data))
    
    instances_pos = np.asarray([instances for temp, instances in sorted(data_pos)])
    instances_neg = np.asarray([instances for temp, instances in sorted(data_neg)])
    instances_pos = np.swapaxes(instances_pos, 0, 2)
    instances_neg = np.swapaxes(instances_neg, 0, 2)
    instances_pos = np.moveaxis(instances_pos, 1, -1)
    instances_neg = np.moveaxis(instances_neg, 1, -1)
            
    X = np.concatenate((instances_neg, instances_pos))
    y = np.concatenate((np.zeros((instances_neg.shape[0])), np.ones(instances_pos.shape[0])))

    return X, y

def seqfile_to_instances(seqfile):
    with open(seqfile, 'r') as _seqfile:
        return _seqfile.read().split('\n')[:-1]


def read_data_channels_for_lstmxlstm_dep(directory, partition, temperatures, categories=["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]):
    """
    Reads csv data to numpy NOT stacking temperatures.

    Args:
        directory (str): Directory containing the files to read.
        partition (str): train, test or val.
        temperatures (list): List of temperatures to read.
        categories (str) : ["OPN", "BUB10", "BUB12", "BUB8", "VRNORM"]

    Returns:
        tuple: (np.array, np.array): X, Y data.
    
    Returned X extra info:
    Size 200 (bases) x 13 (channels)
        Channel 0: OPN probabilities.
        Channel 1: BUB8 probabilities.
        Channel 2: BUB10 probabilities.
        Channel 3: BUB12 probabilities.
        Channel 4: VRNORM probabilities.
        Channels 5, 6, 7, 8: Forward sequence one-hot encoded.
        Channels 9, 10, 11, 12: Reversed sequence one-hot encoded. 
    """

    X = []
    y = []

    instances_pos = []
    instances_neg = []

    for temp in temperatures:
        for tag in ['pos', 'neg']:
            hg = '16' if partition in ['train', 'val'] else '17'
            
            opn_file = directory+'/OPNat'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub8_file = directory+'/BUB8at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub10_file = directory+'/BUB10at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub12_file = directory+'/BUB12at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            vrnorm_file = directory+'/VRNORMat'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            seq_file = directory+'/onlyseq.TSS'+tag+'FineGrained.hg'+hg+'-'+partition+'.'+tag

            opn_data = file_to_array(opn_file)[0]
            bub8_data = file_to_array(bub8_file)[0]
            bub10_data = file_to_array(bub10_file)[0]
            bub12_data = file_to_array(bub12_file)[0]
            vrnorm_data = file_to_array(vrnorm_file)[0]
            seq_data_fw, seq_data_rv = seq_to_onehot_array(seq_file)
            for i in range(len(seq_data_fw)):
                combined_data = [opn_data[i],
                                bub8_data[i],
                                bub10_data[i],
                                bub12_data[i],
                                vrnorm_data[i],
                                *seq_data_fw[i].swapaxes(0, 1),
                                *seq_data_rv[i].swapaxes(0, 1)
                            ]
                X.append(combined_data)
                y.append(1) if tag == 'pos' else y.append(0)
    return np.asarray(X), np.asarray(y)


def read_data_channels_for_lstmxlstm(directory, partition, temperatures, categories=["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]):
    """
    Reads csv data to numpy stacking temperatures.

    Args:
        directory (str): Directory containing the files to read.
        partition (str): train, test or val.
        temperatures (list): List of temperatures to read.
        categories (str) : ["OPN", "BUB10", "BUB12", "BUB8", "VRNORM"]

    Returns:
        tuple: (np.array, np.array): X, Y data.
    
    Returned X extra info:
    Size 28 (temperatures) x 200 (bases) x 13 (channels)
        Channel 1: OPN probabilities.
        Channel 2: BUB8 probabilities.
        Channel 3: BUB10 probabilities.
        Channel 4: BUB12 probabilities.
        Channel 5: VRNORM probabilities.
        Channels 6, 7, 8, 9: Forward sequence one-hot encoded.
        Channels 10, 11, 12, 13: Reversed sequence one-hot encoded. 
    """

    X = []
    y = []

    data_pos = []
    data_neg = []

    for temp in temperatures:
        for tag in ['pos', 'neg']:
            hg = '16' if partition in ['train', 'val'] else '17'
            
            opn_file = directory+'/OPNat'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub8_file = directory+'/BUB8at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub10_file = directory+'/BUB10at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            bub12_file = directory+'/BUB12at'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            vrnorm_file = directory+'/VRNORMat'+temp+'K.hg'+hg+'-'+partition+'.'+tag
            seq_file = directory+'/onlyseq.TSS'+tag+'FineGrained.hg'+hg+'-'+partition+'.'+tag

            opn_data = file_to_array(opn_file)[0]
            bub8_data = file_to_array(bub8_file)[0]
            bub10_data = file_to_array(bub10_file)[0]
            bub12_data = file_to_array(bub12_file)[0]
            vrnorm_data = file_to_array(vrnorm_file)[0]
            seq_data_fw, seq_data_rv = seq_to_onehot_array(seq_file)
            
            combined_data = np.asarray([opn_data,
                                        bub8_data,
                                        bub10_data,
                                        bub12_data,
                                        vrnorm_data,
                                        *[seq_data_fw[:, :, i] for i in range(seq_data_fw.shape[2])],
                                        *[seq_data_rv[:, :, i] for i in range(seq_data_rv.shape[2])]
                                        ])
            
            combined_data = np.moveaxis(combined_data, 0, -1)

            if tag == "pos":
                data_pos.extend(combined_data)
            elif tag == "neg":
                data_neg.extend(combined_data)

    instances_pos = np.asarray(data_pos)
    instances_neg = np.asarray(data_neg)
            
    X = np.concatenate((instances_neg, instances_pos))
    y = np.concatenate((np.zeros((instances_neg.shape[0])), np.ones(instances_pos.shape[0])))

    return X, y

def seqfile_to_instances(seqfile):
    with open(seqfile, 'r') as _seqfile:
        return np.array(_seqfile.read().split('\n')[:-1])


"""
Size 28 (temperatures) x 200 (bases) x 13 (channels)
        Channel 0: OPN probabilities.
        Channel 1: BUB8 probabilities.
        Channel 2: BUB10 probabilities.
        Channel 3: BUB12 probabilities.
        Channel 4: VRNORM probabilities.
        Channels 5, 6, 7, 8: Forward sequence one-hot encoded.
        Channels 9, 10, 11, 12: Reversed sequence one-hot encoded.
"""

def get_seq(X):
    return X[:,0,:,5:9]

def get_reversed_seq(X):
    return X[:,0,:,9:13]

def get_opn_probs(X):
    X = X[:,:,:,0]
    X = np.reshape(X, (*X.shape, 1))
    return X

def get_bub8_probs(X):
    X = X[:,:,:,1]
    X = np.reshape(X, (*X.shape, 1))
    return X

def get_bub10_probs(X):
    X = X[:,:,:,2]
    X = np.reshape(X, (*X.shape, 1))
    return X

def get_bub12_probs(X):
    X = X[:,:,:,3]
    X = np.reshape(X, (*X.shape, 1))
    return X

def get_vrnorm_probs(X):
    X = X[:,:,:,4]
    X = np.reshape(X, (*X.shape, 1))
    return X