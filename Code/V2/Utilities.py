<<<<<<< HEAD:Code/V2/Utilities.py
=======
import csv


def global_variables(af_array, lr_array, dr_array):
    global activation_function_array
    global learning_rate_array
    global dropout_rate_array
    activation_function_array = af_array
    learning_rate_array = lr_array
    dropout_rate_array = dr_array
>>>>>>> master:Code/Utilities.py

def csv_log(file_name, counter, max_acc):
    temp = open(file_name)
    with open(file_name, mode="a+")as file:
        writter = csv.writer(file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

<<<<<<< HEAD:Code/V2/Utilities.py
def global_variables(af_array, lr_array, dr_array):
    global activation_function_array
    global learning_rate_array
    global dropout_rate_array
    activation_function_array = af_array
    learning_rate_array = lr_array
    dropout_rate_array = dr_array
=======
        writter.writerow([str(counter), str(max_acc)])
>>>>>>> master:Code/Utilities.py

    temp.close()

    
def int_to_bit(i):
    if i == 0:
        return "0"
    s = ''
    while i:
        if i & 1 == 1:
            s = "1" + s
        else:
            s = "0" + s
        i //= 2
    return s

def max_bi(binary):
    s = ''
    for i in range(binary):
        s = '1' + s
    return s

def ANN_bit(num, max_num_nodes):
    """
    Parameters                      number of bits
    used                            1
    num_hidden_layers               num
    num_nodes_per_layer             max_num_nodes  >> List num size
    learning_rate                   len(learning_rate)
    activationFunction_per_layer    len(activation_function)
    dropout_rate                    len(dropout_rate)
    """
    used = '1' #The ANN layer will ALWAYS be used!
    #str(random.randint(0,1))
    #print(used)
    num_hidden_layers = int_to_bit(num)

    #print(num_hidden_layers)
    num_nodes_per_layer = ''
    for i in range(num):
        num_nodes_per_layer += int_to_bit(max_num_nodes)

    #print(len(int_to_bit(max_num_nodes)))
    #print(num_nodes_per_layer)
    learning_rate = int_to_bit(len(learning_rate_array)-1)
    #print(learning_rate)
    activationFunction_per_layer = ''
    for i in range(num):
        activationFunction_per_layer += int_to_bit(len(activation_function_array)-1)
    #print(activationFunction_per_layer)

    dropout_rate = int_to_bit(len(dropout_rate_array)-1)
    #print(dropout_rate)
    #exit()
    parameter = used+num_hidden_layers+num_nodes_per_layer+learning_rate+activationFunction_per_layer+dropout_rate
    #print(parameter)
    return parameter

def ANN_bit_to_model(bit, max_num_hidden_layers, max_num_nodes, output_layer):
    """
    Parameters                      starting Bit point                                                                              Ending Bit Point
    used                            0                                                                                               0
    num_hidden_layers               1                                                                                               max_num_hidden_layers
    num_nodes_per_layer             max_num_hidden_layers + 1                                                                       max_num_nodes * max_num_hidden_layers
    learning_rate                   max_num_nodes * max_num_hidden_layers + 1                                                       len(learning_rate) * max_num_nodes * max_num_hidden_layers
    activationFunction_per_layer    len(learning_rate) * max_num_nodes * max_num_hidden_layers + 1                                  len(activation_function) * len(learning_rate) * max_num_nodes * max_num_hidden_layers
    dropout_rate                    len(activation_function) * len(learning_rate) * max_num_nodes * max_num_hidden_layers + 1       len(dropout_rate) * len(activation_function) * len(learning_rate) * max_num_nodes * max_num_hidden_layers
    """
    used = 1
    #print(used)
    #bits 1 > max_num_nodes bit length
    bit_hidden_layers = ''
    bit_length = len(int_to_bit(max_num_hidden_layers))+1
    i = 1

    while i < bit_length:
        bit_hidden_layers += str(bit[i])
        i += 1

    num_hidden_layers = int(bit_hidden_layers, 2)
    if num_hidden_layers == 0:
        num_hidden_layers = 1
    #print(num_hidden_layers)
    num_nodes_per_layer = []

    i = 0
    for i in range(max_num_hidden_layers):
        j = 1
        bit_num_nodes = ''
        while j < (len(int_to_bit(max_num_nodes))+1):
            bit_num_nodes += str(bit[bit_length])

            bit_length += 1
            j+=1

        num_nodes_per_layer.append(int(bit_num_nodes, 2))

    #print("TEST")
    for i in range(len(num_nodes_per_layer)):
        #print(num_nodes_per_layer)
        if num_nodes_per_layer[i] == 0:
            #print(num_nodes_per_layer[i])
            num_nodes_per_layer[i] = 1
<<<<<<< HEAD:Code/V2/Utilities.py


=======


>>>>>>> master:Code/Utilities.py
    if num_hidden_layers == 1:
        num_nodes_per_layer[0] = output_layer
    else:
        num_nodes_per_layer[num_hidden_layers-1] = output_layer
    #print(num_hidden_layers)
    #print(num_nodes_per_layer)
    #bit_length = max_num_hidden_layers * (len(int_to_bit(max_num_nodes))+1)
    prvious_math = bit_length

    #print(len(int_to_bit(len(learning_rate_array)-1)))
    bit_learning_rate = ''
    i = 0
    for i in range(len(int_to_bit(len(learning_rate_array)-1))):
        bit_learning_rate += str(bit[bit_length])
        bit_length += 1

    learning_rate = int(bit_learning_rate, 2)

    i = 0
    for i in range(len(learning_rate_array)):
        if learning_rate == i:
            learning_rate = learning_rate_array[i]

    #print(learning_rate)
    #bit_length = prvious_math + len(int_to_bit(len(activation_function_array))) + 2
    prvious_math = prvious_math + (len(int_to_bit(len(activation_function_array)))*max_num_hidden_layers)

    i = 0
    activation_function_int = []
    for i in range(max_num_hidden_layers):
        bit_activation_function = ''
        j = 0
        while j < len(int_to_bit(len(activation_function_array)-1)):
            bit_activation_function += str(bit[bit_length])
            bit_length += 1
            j+=1
            #print(bit_activation_function)
        activation_function_int.append(int(bit_activation_function, 2))

    i = 0
    activation_function_per_layer = []
    for i in range(max_num_hidden_layers):
        j = 0
        for j in range(len(activation_function_array)):
            if activation_function_int[i] == j:
                activation_function_per_layer.append(activation_function_array[j])

    #print(activation_function_per_layer)
    i = 0
    bit_dropout_rate = ''

    #bit_length = prvious_math + len(int_to_bit(len(dropout_rate_array)))
    prvious_math = bit_length
    #print(bit_length)
    for i in range(len(int_to_bit(len(dropout_rate_array)))-1):
        bit_dropout_rate += str(bit[bit_length])
        bit_length += 1


    dropout_rate = int(bit_dropout_rate, 2)


    i = 0
    for i in range(len(dropout_rate_array)):
        if dropout_rate == i:
            dropout_rate = dropout_rate_array[i]

    #print(dropout_rate)

    #ANN_model = networks.ANN(num_hidden_layers, num_nodes_per_layer, learning_rate,activation_function_per_layer, dropout_rate, used=used )
    return num_hidden_layers, num_nodes_per_layer, learning_rate,activation_function_per_layer, dropout_rate, used
