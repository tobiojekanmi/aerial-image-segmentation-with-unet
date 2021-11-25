# Aerial Image Segmentation With U-Net

##Background:
The usability and performance of an autonomous drone, also known as unmanned aerial vehicle (UAV) are determined by its ability to fly, navigate and land precisely. Current self-navigation systems in most low-altitude autonomous drones rely heavily on the global positioning system (GPS) and sensor information which are of low intelligence and are prone to errors due to signal failures. This can be attributed to the fact that our environment is complex, dynamic, and was designed for human use; hence, they can sometimes be difficult to process without vision. This shortcoming has subsequently amplified the use of computer vision techniques for full drone navigation and landing autonomy. Therefore, this project aims to segment real life aerial images collected by UAVs using state-of-art deep learning image segmentation architectures. This project could serve as a basis for more advanced research work in autonomous drone vision.

##Method:
3,269 pre-labelled aerial images shot at an altitude of about 5 to 50 meters above ground were used to carry out the experiments. The DeepLabV3+ and U-Net segmentation architectures were initially trained and evaluated using the MobileNetV2 model as their encoder/backbone. The model with the superior performance (highest mean intersection-over-union (mIoU)) was then re-trained using additional four encoder/backbone models (DenseNet121, EffecientNetB0, ResNet50 and VGG19) to identify whether the superior computational effectiveness of the MobileNetV2 architecture is also performance-wise justifiable. The segmentation models were trained to classify the aerial images’ pixels into 12 classes — person, bike, car, drone, boat, animal, obstacle, construction, vegetation, road, sky and others/background.

##Results: 
The mean intersection-over-union (mIoU) was used as the optimizing metric, while other factors such as number of model parameters, pixel accuracy, precision, and f1-score were used as satisficing metrics. During the initial training, the U-Net emerged with the highest mIoU of 45% compared to the 37% obtained from the DeepLabV3+ model after training both models for 50 epochs. Also, the mIoU obtained from re-training the U-Net model using additional encoder models are 64%, 57%, 64%, and 62% for DenseNet121, EffecientNetB0, ResNet50, and VGG19 respectively. Overall, using the DenseNet121 with U-Net proved to be the best model after 50 epochs because of superior performance and fair parameters size and could be deep-trained and modified to improve performance.

##Conclusion:
From the results obtained, the U-Net segmentation model seemed to edge the DeepLabV3+ segmentation model while the DenseNet121 classification model seems to outperform other select encoder models as U-Net encoders. However, this cannot be totally relied on because I only ran few epochs because Google Colaboratory only has limited GPU-size and regulated duration. Similarly, improving the model performance to identify and classify all labels properly irrespective of their pixel size using class weighting wasn’t carried out during this project because of the same Google Colaboratory limited and capped GPU-size. Overall, finding additional dataset, deep-training and class weighting would be a standard place to improve model performance, while deploying model to test UAV performance in classifying and making decisions would be a great place to continue this project.
