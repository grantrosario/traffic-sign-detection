import os

answer = input("Which files would you like to run?\n \
                d for detection\n \
                r for recognition\n \
                g for german recognition\n \
                t for transfer recogniton\n \
                a for all\n \
                --> ")

def runner(folderName, fileName):
    os.system('python nets/{}/{}-4.py'.format(folderName, fileName))
    os.system('python nets/{}/{}-6.py'.format(folderName, fileName))
    os.system('python nets/{}/{}-9-1x1.py'.format(folderName, fileName))
    os.system('python nets/{}/{}-9.py'.format(folderName, fileName))
    os.system('python nets/{}/{}-11.py'.format(folderName, fileName))
    os.system('python nets/{}/vote.py'.format(folderName))

if (answer == 'd'):
    runner('detection', 'detect')

elif (answer == 'r'):
    runner('recognition', 'recognize')

elif (answer == 'g'):
    runner('german_recognition', 'recognize')

elif (answer == 't'):
    runner('transfer_recognition', 'recognize')

elif (answer == 'rt'):
    runner('recognition', 'recognize')
    runner('transfer_recognition', 'recognize')

elif (answer == 'a'):
    runner('detection', 'detect')
    runner('recognition', 'recognize')
    runner('german_recognition', 'recognize')
    runner('transfer_recognition', 'recognize')
