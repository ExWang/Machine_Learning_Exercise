import os

name_target2read = 'Label.txt'
name_saveTarget = 'Label_total_reform.txt'


def reform_sth(dir_root, name_file, name_target):
    path_txt_read = dir_root + '\\' + name_file
    path_target_write = name_target
    with open(path_target_write, 'a+') as target_file:

        with open(path_txt_read) as txt_file:

            txt_lines = txt_file.readlines()
            print(len(txt_lines))
            for line in txt_lines:
                line_sp = line.split(' ')
                line_total = ''
                len_line_sp = len(line_sp)
                for i in range(1, len_line_sp):
                    if i == 1:
                        line_total = line_sp[i]
                    if i < len_line_sp-1:
                        line_total = line_total + ' ' + line_sp[i]
                    if i == len_line_sp-1:
                        line_total = line_total + ' ' + line_sp[i].strip('\n') + ';'
                target_file.write(line_total)
    print('Done!')


def work():
    for root, dirs, files in os.walk('./'):

        for file in files:
            if file == name_target2read:
                print('PATH:', root)
                print('Target Detected !')
                reform_sth(root, file, name_saveTarget)
    print('+ALL Done !+')


work()
