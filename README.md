# Training a facial emotion classifier with fast.ai

## Background
This project replicates a Kaggle competition to build a Facial Expression Recognition (FER) model to classify images of faces with emotions using Fast.AI's library. Following the guidelines of the competition, we will use the data provided to categorize each face based on 7 different types of emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral. Our resulting best model achieved a 70.24% accuracy, which came fairly close with the state-of-the-art model (73.28% accuracy). The final model used VGG16 architecture with Cross Entropy Loss, trained on 20 epochs with a learning rate of 0.002.

## Dataset
We are using the FER2013 dataset. We chose to do a Kaggle competition in part because we will have access to many different competition entries so that we can learn how different implementations lead to different results.

## Modeling Setup
After some experiment, we found that using the VGG16_bn architecture with Cross Entropy Loss as our loss function, and using accuracy as our metric. We attempt to use the pre-trained VGG16 model to target our 7 emotions, namely Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.

## Approach
We will attempt to replicate the results from the state-of-the-art FER model using fast.ai's library. We first tried out using different architectures, namely RESNET34, VGG16_BN, VGG16_BN with an SGD optimizer function, as well as a VGG19_BN. We chose the model with the best results and experimented with data augmentation, specifically resizing the images from 48x48 to 160x160. Finally, we experimented the use of DeepAI's Colorized model to colorize our images before training.

## Validation Approach
In this project we use the train/test split validation approach. Since we used a competition dataset, it comes with our training and testing data already split for us, with 28709 images for training, and 7178 for testing.

## Baseline Results
As a baseline, we implemented a RESNET34 architecture using Cross Entropy Loss as our loss function. We used fast.ai's learning rate finder to decide on a learning rate of 0.004, and fine tuned our model with 8 epochs, giving us an accuracy of 0.625801. This was a little far away from our goal of achieving 73.28% accuracy, but it was a good starting point.

## Improving Results
### Experiments with different architectures & optimizer functions
We experimented with the following combinations of architectures, optimizer functions, epochs, and learning rates and achieved the following results:

![image](https://github.com/dkwik/AI-Image-Classification-Neural-Network/assets/89932747/b72fddc3-0d0c-458a-b1c0-e3e24561a9f0)
From our experiments, we determined that VGG16 with Cross Entropy Loss as our loss function was the best setup to train our model

###  Data Augmentation
Next, we used our model architecture and performed some data augmentation on our images. We did this by using fast.ai's default data augmentation algorithm and manually resized the images from 48x48 to 160x160. We retrained the model and got our best accuracy result of 0.702424.

### Final Model
Our final model is fine tuned on vgg16_bn network with cross entropy loss function, using the 160 by 160 pixels augmented images. It is trained for 30 epochs with LR=0.0012 and another 10 epochs with LR=3e-04.

![image](https://github.com/dkwik/AI-Image-Classification-Neural-Network/assets/89932747/5584df2f-e404-415a-8ce7-b3421c75619c)

## Summary
Overall, we found that our best model (70.24% accuracy) that came pretty close to the state-of-the-art model (73.28% accuracy) was the VGG16 architecture, using Cross Entropy Loss, trained on 20 epochs with a learning rate of 0.002. Unsurprisingly, this was the same architecture used for the state-of-the-art model, although they used a more advanced learning rate scheduler.

We believe that while we did not reach state-of-the-art, we achieved our goal of transfer learning and successfully implemented a close model through fast.ai's library. In the future, we would like to continue experimenting with more sophisticated optimizer algorithms using learning rate schedulers.


## Links
- [Project Proposal](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/proposal.ipynb)
- [Final Model](https://github.com/dkwik/CS344_AI_image_classification_model/blob/main/Final%20Model.ipynb)
