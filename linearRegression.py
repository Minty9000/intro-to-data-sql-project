#!/usr/bin/env python

# STEP 1 START
import sys
import os
import importlib
import pathlib
import numpy as np
import zipfile
from io import TextIOWrapper

# expected to be /Algorithms/ScalableAlgorithms/PythonScripts
currentDir = pathlib.Path(__file__).absolute()

parentDir = currentDir.parent.parent.parent
if parentDir not in set(sys.path):
    sys.path.append(str(parentDir))
    print(sys.path)
from ModuleDir import functions

importlib.reload(functions)
dir(functions)
# STEP 1 END

# STEP 2 EDITED IN functions.py

# STEP 3 START
fieldDict = functions.fieldDictBuild()
# STEP 3 END

# STEP 4 START
path = f"{parentDir}/Data/"
fileList = os.listdir(path)
print(parentDir)
print(path)
print(fileList)

# STEP 11 START
q = 4
A = np.zeros(shape=(q, q))
z = np.matrix(np.zeros(shape=(q, 1)))
sumSqrs = 0
n = 0
# STEP 11 END
for filename in fileList:
    try:
        shortYear = int(filename[6:8])
        if shortYear < 11:
            1 / 0
        year = 2000 + shortYear

        file = path + filename
        print(filename)
        fields = fieldDict[shortYear]

        # STEP 5 START
        fEduc = fields["education"]
        sInc, eInc = fields["income"]
        sBMI, eBMI = fields["bmi"]
        fGH = fields["genhlth"]
        # STEP 5 END

        # STEP 6 START
        with zipfile.ZipFile(file) as zf:
            zipFileList = zf.namelist()
            assert len(zipFileList) == 1
            with zf.open(zipFileList[0]) as f:
                reader = TextIOWrapper(f, encoding="latin-1")
                for record in reader:
                    education = functions.getEducation(record[fEduc - 1])
                    income = functions.getIncome(record[sInc - 1 : eInc])
                    bmi = functions.convertBMI(record[sBMI - 1 : eBMI], shortYear)
                    # STEP 6 END

                    # STEP 7 EDITED IN functions.py
                    # STEP 8 EDITED IN functions.py

                    # STEP 9 START
                    y = functions.getHlth(record[fGH - 1])
                    # STEP 9 END

                    # STEP 10 START
                    try:
                        if education < 9 and income < 9 and 0 < bmi < 99 and y != -1:
                            x = np.matrix([1, income, education, bmi]).T
                        else:
                            1 / 0

                        # STEP 12 START
                        A += x * x.T
                        z += x * y
                        sumSqrs += y**2
                        n += 1
                        # STEP 12 END

                        # STEP 13 START
                        if n % 10_000 == 0 and n != 0:
                            b = np.linalg.inv(A) @ z
                            print("\t".join(str(float(bi[0, 0])) for bi in b))
                        # STEP 13 END
                    except ZeroDivisionError:
                        pass
                    # STEP 10 END

    except (ValueError, ZeroDivisionError):
        pass
# STEP 4 END

# STEP 14 START
ybar = z[0,0] / n   # careful: intercept term in z includes Σy
varEst = sumSqrs - n * (ybar**2)
# STEP 14 END

# STEP 15 START
rss = sumSqrs - (b.T @ z)[0,0]   # residual sum of squares
r2 = 1 - rss / varEst
p = q - 1  # number of predictors (exclude intercept)
rAdj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
# STEP 15 END

# STEP 16 START
bList = [str(round(float(bi[0, 0]), 3)) for bi in b]
print('\t'.join(('n', 'b0', 'inc', 'edu', 'bmi', 'r^2_adj')))
print(n, "\t".join(bList), round(float(rAdj), 3), sep="\t")
# STEP 16 END