import os


def countf(dirname, ext, count = 0 ):
    filenames = os.listdir(dirname)
    count_all = 0
#    print(filenames)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            count_all += countf(filepath, ext, count = count_all )
        else:
            if ext == os.path.splitext(filename)[-1]:
                count_all += 1
    return count_all

def wordidwrite(id):
    with open('wordlist.txt', 'a') as wordlist:
        wordlist.write(id+'\n')

def searchwd(dirname, ext, result =[] ):
    filenames = os.listdir(dirname)
#    print(filenames)
    result_all = []
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            result_all += searchwd(filepath, ext)
        else:
            result_all.append(filename.split('_')[0])
#                print result
    result_all = list(set(result_all))
    return result_all

path = '/home/rockheung/acoa_dataset/final_data_v2'
count = countf(path, '.JPEG')
# _RATIO_VALIDATION = 0.2
# _SPLITS_TO_SIZES = {'train': int(round(count*(1-_RATIO_VALIDATION))),
#                     'validation': int(round(count*_RATIO_VALIDATION))}
for wordid in searchwd(path, '.JPEG'):
    wordidwrite(wordid)
#print(_SPLITS_TO_SIZES)
