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

def search(dirname, ext, result =[] ):
    filenames = os.listdir(dirname)
#    print(filenames)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            search(filepath, ext)
        else:
            if ext == os.path.splitext(filepath)[-1]:
                result.append(filepath)
#                print result
    return result

path = '/home/rockheung/acoa_dataset/final_data_v2/Shoe'
count = countf(path, '.JPEG')
_RATIO_VALIDATION = 1 - 0.618
_SPLITS_TO_SIZES = {'train': int(round(count*(1-_RATIO_VALIDATION))),
                    'validation': int(round(count*_RATIO_VALIDATION))}

print(_SPLITS_TO_SIZES)
