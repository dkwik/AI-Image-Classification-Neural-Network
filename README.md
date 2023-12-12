# Facial Expression Recognition 
## Nontechnical Explanation
This project builds an AI image classification model that matches people's faces with one of the seven emotions (angry, disgust, fear, happy, sad, surprise, neutral). The model is trained on 28,709 images of people's faces and their corresponding emotions, and it's tested on 7,178 images that the model has never seen before. 

## Background
The model is built on top of a large pretrained vision model (VGG16). The mdoel was initialized using all of the pretrained model's parameters and fine tuning was done for our specific use case of facial recognition. And as the model gets trained, it picks up specific characteristics of the facial images gets better at classifying what emotion is presented in a given facial image. 

## Results
The final model achieves 70% accuracy in the testing set, which is very close to the state-of-the-art performance of 73%. Estimated human performance is about 65.5%.

## Links
- [Project Proposal](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/proposal.ipynb)
- [Final Model](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/Final%20Model.ipynb)
- [Report](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/Report.ipynb)
