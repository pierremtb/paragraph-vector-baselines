# Grabs data from files using helper functions
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 10 2019

import os

def sortFile(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    return int(basename)

class DataGrabber:

    def __init__(self, path):
        self.path = path

    def readCommentFiles(self):
        comments = []
        for filename in sorted(os.listdir(self.path), key=sortFile):
            try:
                with open(self.path + filename, encoding='ISO-8859-1') as file:
                    comments.append(file.readline())
            except UnicodeDecodeError:
                print(filename + " failed")

        return comments



