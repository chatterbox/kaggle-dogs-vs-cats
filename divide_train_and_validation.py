import os
import shutil

if __name__ == "__main__":
    BASE_DIR = os.path.abspath('.')
    val_folder = 'validation'
    IMG_train = 'train'  # the folder where preserve the train image
    IMG_all = 'all(train and val)'  # the folder where preserve the train image

    IMG_test = 'test'  # the folder where preserve the test image
    path_all = os.path.join(BASE_DIR, IMG_all)
    path_train = os.path.join(BASE_DIR, IMG_train)
    path_test = os.path.join(BASE_DIR, IMG_test)
    val_path = os.path.join(BASE_DIR, val_folder)
    all_num = 0

    val_num_flag = 500
    val_num = 500
    print(BASE_DIR)

    for files in os.listdir(BASE_DIR):      # if there is a validation file in the root file,delete the old one and create a new one
        print(files)
        if files == val_folder:
            print("the validation folder is already exist")
            shutil.rmtree(val_path, True)
        if files == IMG_train:
            print("the train folder is already exist")
            shutil.rmtree(path_train, True)

    os.makedirs(val_folder)
    os.makedirs(IMG_train)


    for files in os.listdir(path_all):
        all_num = all_num + 1
        if val_num > 0:
            path_all_file = os.path.join(path_all, files)
            shutil.copy(path_all_file, val_path)
        val_num = val_num - 1


    flag = 0

    for files in os.listdir(path_all):
        if flag >= val_num_flag:
            path_all_file = os.path.join(path_all, files)
            shutil.copy(path_all_file, path_train)
        flag = flag + 1

