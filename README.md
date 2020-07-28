# MINST-handwritten-image-recognizer

So the folowing repo is a basic application of neural networks for image classification
In the code we use the MINST dataset provided by the famous Keras API. The dataset consits of hand witten grey scale images of numbers from 0-9, Our task is to make a neural network model to classify these images for the same
 
In the repo the First file Digitrecognizer.py does the above task using a simple ANN(artificial neural networks). if you ran the file the model does really well on the training set but fails on teh test set and results in less accuracy. You can play around with th enumbers and see for yourself if the accuracy improves or no. It depends you might get a great accuracy but i didn't get.

The second file in the repo number_classifier.py or you can also use the .ipynb file as well. The model is a CNN (convolutional neural networks). The model is build on the famous LeNet5 architecture. The model Not only does well on the trian set but also does really well on the test set as well. When I trained the modle I got and accuracy of approximately 99%. Again you can play around with the numbers to get more acuuracy if you wish 
