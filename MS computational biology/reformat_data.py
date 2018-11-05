import sys
from string import *


def main(argv):
    data_path = argv[1]
    new_filename = argv[2]

    read_data_from_file(data_path, new_filename)


def read_data_from_file(path, new_filename):
    filename = path
    print filename

    data = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            row = line.split(' ')

            row[1] = row[1].replace("qid:WT04-", '')

            for i in xrange(len(row)):
                column = row[i]
                index_of_colon = column.find(':')
                if index_of_colon != -1:
                    row[i] = row[i][index_of_colon + 1:]

            row[-1] = row[-1].replace("\r\n", '')
            row.remove("#docid")
            row.remove("=")

            hexnum = string_to_hex(row[-1])
            row[-1] = hexnum
            row.insert(-1, row.pop(1))

            row_string = ""

            for column in row:
                row_string = row_string + ',' + str(column)

            data.append(row_string)

            line = fp.readline()

    with open(new_filename, "a") as new_file:
        for row in data:
            new_file.write(row)
            new_file.write('\n')


def string_to_hex(s):
    lst = []
    for ch in s:
        hv = hex(ord(ch)).replace('0x', '')
        if len(hv) == 1:
            hv = '0' + hv
        lst.append(hv)

    return int(reduce(lambda x, y: x + y, lst), 16)


def hex_to_string(s):
    if not isinstance(s, basestring):
        string = hex(s).rstrip("L").lstrip("0x")
    else:
        string = s
    return string and chr(atoi(string[:2], base=16)) + hex_to_string(string[2:]) or ''


if __name__ == '__main__':
    main(sys.argv)
