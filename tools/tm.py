import os
root = '/home/yl/python/AdelaiDet1/tools/we_test/'
for i in os.listdir(root + '15'):
    for j in os.listdir(root + '16'):
        if i ==j:
            print(i)