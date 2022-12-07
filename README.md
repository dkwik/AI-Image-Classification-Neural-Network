# CS 344 Final Project - Facial Expression Recognition Challenge
## Nontechnical Explanation
This project is to build an AI image classification model that matches people's faces with one of the seven emotions (angry, disgust, fear, happy, sad, surprise, neutral). The model is trained on 28,709 images of people's faces and their corresponding emotions, and it's tested on 7,178 images that the model has never seen before. Our model is built on top of a large pretrained vision model (VGG16) by initializing our model using all of the pretrained model's parameters and fine tuning them to match our specific use case. This way, the model has an "idea" of what a general image consists of from the beginning of the training. And as the model gets trained, it picks up specific characteristics of the facial images and fine tunes our complex mathematical model to better classify what emotion is presented in each image. Our final model achieves 70% accuracy in the testing set, which is very close to the state-of-the-art performance of 73%, whereas the estimated human performance is only about 65.5%.

## Links
- [Project Proposal](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/proposal.ipynb)
- [Final Model](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/Final%20Model.ipynb)
- [Report](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/Report.ipynb)
