#-*-coding:utf-8-*-
import os, math, collections
import numpy as np


def trav_dir(p):
    file_list = []
    for root, dirs, files in os.walk(p):
        for f in files:
            per_path = str(root + '\\' + f)
            file_list.append(per_path)
    return file_list

def get_data(d_p):
    label_vec = []
    files = trav_dir(d_p)
    for f in files:
        f_s = f.split('\\')
#        print("f.split('\\'): ", f_s)
        per_label_vec = []
        per_label = int(f_s[-1][0])  # each vector's label
#        print("per_label: ", per_label)
        per_vec = ((np.loadtxt(f)).ravel())  # each file's vector
#        print("per_vec: ", per_vec)
        per_label_vec.append(per_label)  # 将一个文件的标签和向量放到同一个list内 (连着下行)
        per_label_vec.append(per_vec)  # 目的是将标签和向量对应起来,类似于字典,这里不直接用字典因为字典的键不可重复。
        label_vec.append(per_label_vec)
    return label_vec

def write_cnt(pth, cnt, c_to_enode):  # https://stackoverflow.com/questions/10971033/backporting-python-3-openencoding-utf-8-to-python-2
    with open(pth, 'a', encoding=c_to_enode) as f:
        f.write(cnt)
    return pth + ' is updated!'


def read_cnt(pth, c_encoded):
    with open(pth, 'r', encoding=c_encoded) as f:
        lines = f.readlines()
    return lines


def change_data(list_of_file_names, path_to_input):
    train_path = make_dir(path_to_input+'\\'+'trainingDigits')
    test_path = make_dir(path_to_input+'\\'+'testDigits')
    for f in list_of_file_names:
        per_name = f.split('\\')[-2:]  # with -2, we include file's parent dir
        new_path = path_to_input+'\\'+'\\'.join(per_name)
        per_f_cnt = read_cnt(f, 'utf8')
        new_f_cnt = []
        for per_line in per_f_cnt:
            line_sgl_chr_seq = list(per_line.replace('\n', '').replace('\r', ''))  # unravel a whole str into individual chrs
            new_f_cnt.append(' '.join(line_sgl_chr_seq))
        print(write_cnt(new_path, '\n'.join(new_f_cnt), 'utf8'))


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())


def order_dic(dic, flag):
    ordered_list = sorted(dic.items(), key=lambda item : item[1], reverse=flag)
    return ordered_list


def find_label(train_vec_list, vec, k):
    get_label_list = []
    for per_trainlabel_vec in train_vec_list:
        per_label_distance = []
        train_label, train_vec = per_trainlabel_vec[0], per_trainlabel_vec[1]
        vec_distance = Euclidean(train_vec, vec)  # calc Euclidean distance
        per_label_distance.append(train_label)
        per_label_distance.append(vec_distance)  # put lbl & its corresponding dist into list
        get_label_list.append(per_label_distance)
    result_k = np.array(get_label_list)
    order_distance = (result_k.T)[1].argsort()  # sort dists
    order = np.array((result_k[order_distance].T)[0])
    top_k = np.array(order[:k], dtype=int)  # 获取前k距离和标签
    find_label = order_dic(collections.Counter(top_k), True)[0][0]  # 统计在前k排名中标签出现频次
    return find_label


def classify(train_vec_list, test_vec_list, k):
    error_counter = 0
    for per_label_vec in test_vec_list:
        label, vec = per_label_vec[0], per_label_vec[1]
        get_label = find_label(train_vec_list, vec, k)  # 获得学习得到的标签
        print('Original label is:' + str(label) +
              ', kNN label is:' + str(get_label))
        if str(label) != str(get_label):
            error_counter += 1
        else:
            continue
    true_probability = str(round((1 - error_counter/len(test_vec_list)) * 100, 2)) + '%'
    print('Correct probability:' + true_probability)


def main():
    pending_path =r'digits'
    post_path = make_dir(r'.\processed\input_digits')  # omitting r may cause incomplete splits
    files = trav_dir(pending_path)
    change_data(files, post_path)

    k = 3
    train_data_path = post_path + '\\trainingDigits'
    test_data_path = post_path + '\\testDigits'
    train_vec_list = get_data(train_data_path)
    test_vec_list = get_data(test_data_path)
    classify(train_vec_list, test_vec_list, k)


if __name__ == '__main__':
    main()
