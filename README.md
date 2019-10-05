# kaggle-dogs-vs-cats
My job contain two python files: main2.py    and   use_cnnmodel_totest.py

main2.py can train a cnn model to divide dogs and cats images.

use_cnnmodel_totest.py can use the trained cnn model to test the unlabeled images


How to use the two python files?
Here comes the answer:

1.download kaggle Dogs vs Cats dataset(https://www.kaggle.com/c/dogs-vs-cats/data)

2.unzip it all,you will also see two folders(test1 and train),test1 contains the unlabeled images,and train contains 12500 cats images and 12500 dogs images that already labeled .

3.make the (test1 and train) folders and the py files be the same filder called Project

4.Creat a test folder in the Project folder to contain the test1 folder(maybe this is the reason that it is named test1,cause we need to creat a new one)

5.Run the main2.py and you will get a model.pt in your Project folder,model.pt is the trained cnn .

6.Run the use_cnnmodel_totest.py and you can see the ./Project/test/result,in this folders,you can see a lot of the processed images which is already be put a label on.(on the center of the images,"Dog" or "Cat")

Tips: I just used a random cnn model to train,so the accuracy is low(only 76%),so if you are not satisfied with the cnn,you can easily change the cnn model for a better result.(maybe a vgg or resnet)
