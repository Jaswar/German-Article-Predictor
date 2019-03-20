# German-Article-Predictor

Description:
  As the name suggests this is a repository with a code that predicts what article a german word is. Obviously what I could do is to just save entire dictionary of words and read it every time. This was not my aim, however. What I wanted to do is to see how random german articles indeed are and therefore see if I could build a classifier for them. So what I did is I created a neural network to do it. This repository contains a Convolutional Neural Network that reaches around 96% accuracy. I got my words' dictionary from a similar GitHub project: https://github.com/aakhundov/deep-german

Required libraries:
- tensorflow (version 1.12.0, recommended to use tensorflow-gpu and theano)
- keras (version 2.2.2)
- numpy (version 1.15.1)
- pandas (version 0.23.4)

My specs:
- i7-7700HQ Processor
- 16GB RAM
- NVIDIA Quadro M1200 4GB graphics card

Training:
  What my code does first is it converts all words to OneHotEncoder form by changing every letter to a string of zeros of alphabet's length and one in place where this letter would be placed in alphabet. I do it for every letter and then I just connect them to one long string of ones and zeros. This final string has to be of some constant size, so for every word I add zeros so that they will all fit the fixed size. I also have an array that contains what article every word has. I then train my neural network on these two sets, paramaters such as learning rate and batch size can be easily changed. As mentioned before my CNN reached accuracy of around 96%, meaning that after training it can get most of the articles right. After training you can test your neural network by changing 'train' variable to False and setting neural network file location in 'filepathToOpen'. Then you will be able to type words in console to check if it works (use german characters and only small letters).
