import glob, os, shutil, sys, random

NUMBERSAMPLES = 210
PATH_SOURCE = "/home/smu/Desktop/RNN/audiodata/own_sixseconds/"
PATH_DESTINATION = "/home/smu/Desktop/RNN/audiodata/temp/"

for x in range(NUMBERSAMPLES):
    random_file=random.choice(os.listdir(PATH_SOURCE))
    src = PATH_SOURCE + random_file
    dst = PATH_DESTINATION + random_file
    shutil.move(src,dst)
