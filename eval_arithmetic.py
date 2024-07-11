import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pandas.errors import SettingWithCopyWarning
import warnings
import random

def parse_equation(equation):
    """Parse a single equation 'a+b=c' string into components 'a+b=' and 'c'."""
    try:
        parts = equation.split('=')
        result = parts[1]
        operands = parts[0].strip("\n") + "="
        return operands, result
    except Exception as e:
        print(f"parse_equation has encountered Error: {e} at {equation}")
        raise ValueError(f"parse_equation has encountered Error: {e} at {equation}")

def parse_operands(start):
    """Parse a single string 'a+b' into int components a, b"""
    operands = start.split('+')
    a = int(operands[0])
    b = int(operands[1][:-1])
    return a, b
    
def modular_gtc_ndigit(start, ndigit=3):
    a, b = parse_operands(start)
    modular_c = (a%10**ndigit) + (b%10**ndigit)
    modular_c_reversed_str = str(modular_c)[::-1]
    return modular_c_reversed_str
    
def result_between_first_equal_and_semicolon(input_string):
    """Extract the result part of the equation between the first '=' and ';'."""
    equal_pos = input_string.find('=')
    semicolon_pos = input_string.find(';')
    
    if equal_pos != -1 and semicolon_pos != -1 and equal_pos < semicolon_pos:
        output_string = input_string[equal_pos + 1:semicolon_pos].strip()
    else:
        output_string = None
    
    return output_string

def result_between_equal_and_semicolon(input_string):
    """Extract the result part of the equation between '=' and ';'."""
    equal_pos = input_string.find('=')
    semicolon_pos = input_string.find(';')
    
    if equal_pos != -1 and semicolon_pos != -1:
        output_string = input_string[equal_pos + 1:semicolon_pos]
    else:
        output_string = None
    
    return output_string

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    data_list = data.split(";")
    return data_list

def get_eval_data(data_dir, data_name='input.txt', split='train', n_eval=10000):
    input_file_path = os.path.join(data_dir, data_name)
    data_list = load_dataset(input_file_path)
    
    n = len(data_list)
    if split == 'train':
        eval_data = data_list[:int(n * 0.9)]
    elif split == 'test':
        eval_data = data_list[int(n * 0.9):]
    else:
        eval_data = data_list
    if n_eval == -1:
        print(f"logging: {data_name} not truncated")
    elif len(eval_data) > n_eval:
        eval_data = eval_data[:n_eval]
        print(f"logging: {data_name} truncated to {n_eval} obvs")
    
    return eval_data

def estimate_accuracy(model, eval_data, data_dir, device, max_new_tokens=23, batch_size=128, verbose=False, prepend=True):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        print(f"Warning: meta_path: {meta_path} did not exist")
        raise ValueError(f"Warning: meta_path: {meta_path} did not exist")
    
    stoi, itos = meta['stoi'], meta['itos']
    if not prepend:
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        encode = lambda s: [stoi['<bos>']] + [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l if i not in [stoi['<bos>'], stoi['<eos>']]])
    
    num_samples = 1
    temperature = 1.0
    top_k = 200

    total_data = len(eval_data)
    correct = 0
    total_eval = 0
    all_input_ids = []
    all_ground_truths = []

    for line in eval_data[:-1]:
        try:
            start, gt_c_reversed = parse_equation(line)
            start_ids = encode(start)
            all_input_ids.append(start_ids)
            all_ground_truths.append(gt_c_reversed)
        except Exception as e:
            print(f"Exception {e} happened at line: {line}")
        
    model.eval()
    if verbose:
        pbar = tqdm(total=len(all_input_ids), desc="Evaluating")
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size]
            batch_ground_truths = all_ground_truths[i:i + batch_size]
            batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
            x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
                y = y.to('cpu')
                for j, y_gen in enumerate(y):
                    y_gen_decoded = decode(y_gen.tolist())
                    gen_c_reversed = result_between_first_equal_and_semicolon(y_gen_decoded)
                    if gen_c_reversed is not None and not gen_c_reversed.startswith("\n"):
                        total_eval += 1
                        if batch_ground_truths[j] == gen_c_reversed:
                            correct += 1
            
            pbar.update(len(batch_input_tensors))
            pbar.set_postfix({'Correct': correct, 'Total': total_eval, 'gen_c':gen_c_reversed, 'ground_truth':batch_ground_truths[j]})
        pbar.close()
    else:
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size]
            batch_ground_truths = all_ground_truths[i:i + batch_size]
            batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
            x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
                y = y.to('cpu')
                for j, y_gen in enumerate(y):
                    y_gen_decoded = decode(y_gen.tolist())
                    gen_c_reversed = result_between_equal_and_semicolon(y_gen_decoded)
                    if gen_c_reversed is not None and not gen_c_reversed.startswith("\n"):
                        total_eval += 1
                        if batch_ground_truths[j] == gen_c_reversed:
                            correct += 1
    model.train()
    accuracy = correct / total_eval if total_eval != 0 else 0
    return accuracy


def estimate_accuracy_inspection(model, eval_data, data_dir, device, max_new_tokens=5, batch_size=128, modular=False):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        print(f"Warning: meta_path: {meta_path} did not exist")
        raise ValueError(f"Warning: meta_path: {meta_path} did not exist")
    
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    num_samples = 1
    max_new_tokens = max_new_tokens  # 5 or 6 for 3-digit addition >=n+1+1=5 !!! 8 or 9 for 456 training!!!
    temperature = 1.0
    top_k = 200
    
    correct = 0
    total_eval = 0
    all_input_ids = []
    all_ground_truths = []

    for line in eval_data[:-1]:
        try:
            start, gt_c_reversed = parse_equation(line)
            start_ids = encode(start)
            all_input_ids.append(start_ids)
    
            if not modular:
                all_ground_truths.append(gt_c_reversed)
            else:
                modular_c_reversed_str = modular_gtc_ndigit(start, ndigit=3)
                all_ground_truths.append(modular_c_reversed_str)
        except Exception as e:
            print(f"Exception {e} happened at line: {line}")
            #raise ValueError(f"Exception at {line}")
    gen_pred = []
    trans_pred = []
    label = []
    model.eval()
    pbar = tqdm(total=len(all_input_ids), desc="Evaluating")
    for i in range(0, len(all_input_ids), batch_size):
        batch_input_ids = all_input_ids[i:i + batch_size]
        batch_ground_truths = all_ground_truths[i:i + batch_size]
        batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
        x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
            y = y.to('cpu')
            for j, y_gen in enumerate(y):
                y_gen_decoded = decode(y_gen.tolist())
                gen_c_reversed = result_between_first_equal_and_semicolon(y_gen_decoded)
                gen_pred.append(y_gen_decoded)
                trans_pred.append(gen_c_reversed)
                label.append(batch_ground_truths[j])
                if gen_c_reversed is not None and not gen_c_reversed.startswith("\n"):
                    total_eval += 1
                    if batch_ground_truths[j] == gen_c_reversed:
                        correct += 1
        
        pbar.update(len(batch_input_tensors))
        pbar.set_postfix({'Correct': correct, 'Total': total_eval, 'gen_c':gen_c_reversed, 'ground_truth':batch_ground_truths[j]})
    pbar.close()
    model.train()
    
    accuracy = correct / total_eval if total_eval != 0 else 0
    return accuracy, gen_pred, trans_pred, label


def get_the_last_non_zero_index(string):
    for i in range(len(string)-1, -1, -1):
        if string[i] != '0':
            return i
    return len(string)-1


def get_digit_dummy_acc(gen_c, gt):
    last_valid_digit = get_the_last_non_zero_index(gt)
    gen_last_valid_digit = get_the_last_non_zero_index(gen_c)
    gen_c_r = gen_c[::-1]
    gt_r = gt[::-1]
    digit_result = {}
    for i in range(1, max(gen_last_valid_digit, last_valid_digit)+2):
        if gen_c_r[-i] == gt_r[-i]:
            #print(gen_c_r[-i], gt_r[-i])
            digit_result[i] = 1
        else:
            digit_result[i] = 0
    return digit_result

def estimate_accuracy_digit_by_digit(model, eval_data, data_dir, device, max_new_tokens=23, batch_size=128, verbose=False, prepend=True, debug=False):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        print(f"Warning: meta_path: {meta_path} did not exist")
        raise ValueError(f"Warning: meta_path: {meta_path} did not exist")
    
    stoi, itos = meta['stoi'], meta['itos']
    if not prepend:
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        encode = lambda s: [stoi['<bos>']] + [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l if i not in [stoi['<bos>'], stoi['<eos>']]])
    
    num_samples = 1
    temperature = 1.0
    top_k = 200

    total_data = len(eval_data)
    correct = 0
    total_eval = 0
    all_input_ids = []
    all_ground_truths = []

    for line in eval_data[:-1]:
        try:
            start, gt_c_reversed = parse_equation(line)
            start_ids = encode(start)
            all_input_ids.append(start_ids)
            all_ground_truths.append(gt_c_reversed)
        except Exception as e:
            print(f"Exception {e} happened at line: {line}")
    dig_result_list = []
    model.to(device)
    model.eval()
    if verbose:
        pbar = tqdm(total=len(all_input_ids), desc="Evaluating")
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size]
            batch_ground_truths = all_ground_truths[i:i + batch_size]
            batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
            x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
            x = x.to(device)
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
                x = x.to('cpu')
                y = y.to('cpu')
                for j, y_gen in enumerate(y):
                    y_gen_decoded = decode(y_gen.tolist())
                    gen_c_reversed = result_between_equal_and_semicolon(y_gen_decoded)
                    try:
                        dig_result_temp = get_digit_dummy_acc(gen_c_reversed, batch_ground_truths[j])
                        dig_result_list.append(dig_result_temp)
                        if gen_c_reversed is not None and not gen_c_reversed.startswith("\n"):
                            total_eval += 1
                            if batch_ground_truths[j] == gen_c_reversed:
                                correct += 1
                                if debug:
                                    all_digit_correct = 1
                                    for k, v in dig_result_temp.items():
                                        if v == 0:
                                            all_digit_correct = 0
                                    if not all_digit_correct:
                                        print("generated: {}; groundtruth: {}".format(gen_c_reversed, batch_ground_truths[j]))
                            else:
                                if debug:
                                    all_digit_correct = 1
                                    for k, v in dig_result_temp.items():
                                        if v == 0:
                                            all_digit_correct = 0
                                    if all_digit_correct:
                                        print("generated: {}; groundtruth: {}".format(gen_c_reversed, batch_ground_truths[j]))
                    except Exception as e:
                        if verbose:
                            print(f"{e} was raised at y_gen_decoded")
            pbar.update(len(batch_input_tensors))
            pbar.set_postfix({'Correct': correct, 'Total': total_eval, 'gen_c':gen_c_reversed, 'ground_truth':batch_ground_truths[j]})
        pbar.close()
    else:
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size]
            batch_ground_truths = all_ground_truths[i:i + batch_size]
            batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
            x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
            dig_result_list = []
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
                y = y.to('cpu')
                for j, y_gen in enumerate(y):
                    y_gen_decoded = decode(y_gen.tolist())
                    gen_c_reversed = result_between_first_equal_and_semicolon(y_gen_decoded)
                    try:
                        dig_result_temp = get_digit_dummy_acc(gen_c_reversed, batch_ground_truths[j])
                        dig_result_list.append(dig_result_temp)
                        if gen_c_reversed is not None and not gen_c_reversed.startswith("\n"):
                            total_eval += 1
                            if batch_ground_truths[j] == gen_c_reversed:
                                correct += 1
                    except Exception as e:
                        pass
                        
    model.train()
    dig_result_df = pd.DataFrame(dig_result_list)
    dig_acc = {i:dig_result_df[i].mean() for i in dig_result_df.columns}
    accuracy = correct / total_eval if total_eval != 0 else 0
    return accuracy, dig_acc


import random

def get_digit_dummy_data(gen_c, gt, a, b):
    last_valid_digit = get_the_last_non_zero_index(gt)
    gen_last_valid_digit = get_the_last_non_zero_index(gen_c)
    gen_c_r = gen_c[::-1]
    gt_r = gt[::-1]
    digit_result = {}
    for i in range(1, max(gen_last_valid_digit, last_valid_digit)+2):
        if gen_c_r[-i] == gt_r[-i]:
            #print(gen_c_r[-i], gt_r[-i])
            digit_result[f"acc_{i}"] = 1
            digit_result[f"a_{i}"] = a[-i]
            digit_result[f"b_{i}"] = b[-i]
            digit_result[f"c_{i}"] = gen_c_r[-i]
        else:
            digit_result[f"acc_{i}"] = 0
            digit_result[f"a_{i}"] = a[-i]
            digit_result[f"b_{i}"] = b[-i]
            digit_result[f"c_{i}"] = gen_c_r[-i]
    return digit_result

def random_impute_number(input_, impute_digit_list, impute_number):
    input_list = list(input_)
    inpute_index_list = random.sample(impute_digit_list, impute_number)
    for index in inpute_index_list:
        while True:
            new_digit = str(random.choice([i for i in range(10) if i != int(input_list[-index])]))
            if new_digit != input_list[-index]:
                break
        input_list[-index] = new_digit
    return ''.join(input_list)

def get_padded_result(a, b):
    reversed_result =str(int(a)+int(b))[::-1]
    return reversed_result+(20-len(reversed_result))*'0'

def collect_data_zeta_chi(model, eval_data, data_dir, device, max_new_tokens=23, batch_size=128, verbose=False, prepend=True, random_impute=True, impute_side=1, impute_repeat_num=4, impute_digit_list=[4,5,6], impute_number=2, debug=False):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        print(f"Warning: meta_path: {meta_path} did not exist")
        raise ValueError(f"Warning: meta_path: {meta_path} did not exist")
    
    stoi, itos = meta['stoi'], meta['itos']
    if not prepend:
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        encode = lambda s: [stoi['<bos>']] + [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l if i not in [stoi['<bos>'], stoi['<eos>']]])
    
    num_samples = 1
    temperature = 1.0
    top_k = 200
    
    total_data = len(eval_data)
    all_input_ids = []
    all_ground_truths = []
    
    for line in eval_data[:-1]:
        try:
            start, gt_c_reversed = parse_equation(line)
            start_ids = encode(start)
            all_input_ids.append(start_ids)
            all_ground_truths.append(gt_c_reversed)
            if random_impute:
                for times in range(impute_repeat_num):
                    if impute_side == 1:
                        a = random_impute_number(start.strip('=').split('+')[0], impute_digit_list, impute_number)
                        b = start.strip('=').split('+')[1]
                        start = a + "+" + b + '='
                    elif impute_side == 2:
                        a = start.strip('=').split('+')[0]
                        b = random_impute_number(start.strip('=').split('+')[1], impute_digit_list, impute_number)
                        start = a + "+" + b + '='
                    else:
                        a = random_impute_number(start.strip('=').split('+')[0], impute_digit_list, impute_number)
                        b = random_impute_number(start.strip('=').split('+')[1], impute_digit_list, impute_number)
                        start = a + "+" + b + '='
                    start_ids = encode(start)
                    all_input_ids.append(start_ids)
                    result_reversed = get_padded_result(a, b)
                    all_ground_truths.append(result_reversed)
        except Exception as e:
            print(f"Exception {e} happened at line: {line}")
    digit_dummy_data_list = []
    model.to(device)
    model.eval()
    if verbose:
        pbar = tqdm(total=len(all_input_ids), desc="Evaluating")
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size]
            batch_ground_truths = all_ground_truths[i:i + batch_size]
            batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
            x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
            x = x.to(device)
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
                x = x.to('cpu')
                y = y.to('cpu')
                for j, y_gen in enumerate(y):
                    y_gen_decoded = decode(y_gen.tolist())
                    gen_c_reversed = result_between_equal_and_semicolon(y_gen_decoded)
                    a, b = y_gen_decoded.split("=")[0].split('+')
                    try:
                        digit_dummy_data_temp = get_digit_dummy_data(gen_c_reversed, batch_ground_truths[j], a, b)
                        digit_dummy_data_list.append(digit_dummy_data_temp)
                    except Exception as e:
                        if verbose:
                            print(f"Warning: {e} was raised at {y_gen_decoded}")
            pbar.update(len(batch_input_tensors))
            pbar.set_postfix({'gen_c':gen_c_reversed, 'ground_truth':batch_ground_truths[j]})
        pbar.close()
    else:
        for i in range(0, len(all_input_ids), batch_size):
            batch_input_ids = all_input_ids[i:i + batch_size]
            batch_ground_truths = all_ground_truths[i:i + batch_size]
            batch_input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in batch_input_ids]
            x = torch.nn.utils.rnn.pad_sequence(batch_input_tensors, batch_first=True, padding_value=stoi['\n'])
            x = x.to(device)
            with torch.no_grad():
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, do_sample=False)
                x = x.to('cpu')
                y = y.to('cpu')
                for j, y_gen in enumerate(y):
                    y_gen_decoded = decode(y_gen.tolist())
                    gen_c_reversed = result_between_first_equal_and_semicolon(y_gen_decoded)
                    a, b = y_gen_decoded.split("=")[0].split('+')
                    try:
                        digit_dummy_data_temp = get_digit_dummy_data(gen_c_reversed, batch_ground_truths[j], a, b)
                        digit_dummy_data_list.append(digit_dummy_data_temp)
                    except:
                        #print(y_gen_decoded)
                        pass
    model.train()
    digit_dummy_df = pd.DataFrame(digit_dummy_data_list)
    return digit_dummy_df

def transform_dig_dummy_data(digit_dummy_data):
    # Calculate the number of digits
    digits_len = len(list(digit_dummy_data.columns)) // 4
    
    # Initialize an empty list to hold the transformed rows
    transformed_data = []
    
    # Loop through each row of the DataFrame
    for index, row in digit_dummy_data.iterrows():
        # Loop through each digit (except the first one)
        for i in range(2, digits_len + 1):
            # Define the column names for the current digit and the previous one
            acc_i = f'acc_{i}'
            a_i = f'a_{i}'
            b_i = f'b_{i}'
            c_i = f'c_{i}'
            a_i_minus_1 = f'a_{i-1}'
            b_i_minus_1 = f'b_{i-1}'
    
            # Create a dictionary for the current row
            transformed_row = {
                'acc_i': row[acc_i],
                'c_i': row[c_i],
                'a_i': row[a_i],
                'b_i': row[b_i],
                'a_i-1': row[a_i_minus_1],
                'b_i-1': row[b_i_minus_1],
                'Label_i': i
            }
            # Append the transformed row to the list
            transformed_data.append(transformed_row)
    
    # Convert the list of transformed rows to a DataFrame
    transformed_df = pd.DataFrame(transformed_data)
    return transformed_df


def get_dig_dummy_r2(data, regressor, rep=1):
    warnings.simplefilter('ignore', SettingWithCopyWarning)
    # Ensure that the DataFrame is not empty
    if data.empty:
        return {}
    
    # Convert non-numeric columns to numeric if possible
    for col in ['a_i', 'b_i', 'a_i-1', 'b_i-1']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop rows with NaN values
    data_clean = data.dropna(subset=['a_i', 'b_i', 'a_i-1', 'b_i-1', 'c_i'])
    
    # Calculate 'remain_part' and 'increment_part'
    data_clean['remain_part'] = (data_clean['a_i'] + data_clean['b_i']) % 10
    data_clean['increment_part'] = (data_clean['a_i-1'] + data_clean['b_i-1']) // 10
    
    # Initialize an empty dictionary to store R2 values
    r2_values = {}
    
    # Get unique Label_i values
    unique_labels = data_clean['Label_i'].unique()
    
    # Loop through each unique Label_i
    for label in unique_labels:
        # Filter data for the current label
        label_data = data_clean[data_clean['Label_i'] == label]
        if rep == 1:
            # Prepare features and target
            X = label_data[['remain_part', 'increment_part']]
        elif rep == 2:
            X = label_data[['a_i', 'b_i', 'a_i-1', 'b_i-1']]
        else:
            raise ValueError(f"Parameter rep {rep} is not set to a meaningful value")
        y = label_data['c_i']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the regressor on the training data
        regressor.fit(X_train, y_train)
        
        # Predict on the training and testing sets
        y_train_pred = regressor.predict(X_train)
        y_test_pred = regressor.predict(X_test)
        
        # Calculate R2 for in-sample and out-of-sample predictions
        r2_in_sample = r2_score(y_train, y_train_pred)
        r2_out_of_sample = r2_score(y_test, y_test_pred)
        
        # Store the R2 values in the dictionary
        r2_values[label] = {'R2_in_sample': r2_in_sample, 'R2_out_of_sample': r2_out_of_sample}
    
    return r2_values