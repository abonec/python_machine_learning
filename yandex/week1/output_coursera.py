import os
DIR = 'coursera_out'
def output(file, string):
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    path = os.path.join(DIR, file)
    f = open(path, 'w')
    f.write(string)
    f.close()
