# SLO-MSNet
For the first time, we utilized SLO images to differentiate between MS and HC eyes, with promising results achieved using combination of designed CAE and MLP which we named SLO-MSNet.
The capability of the proposed CAE in detecting more informative representations could be attributed, at least partly, to the connections between two of the encoder and decoder blocks. This will propagate the information between encoder and decoder parts and prevent the precise information to be lost during the up-sampling process, since a feature map with higher resolution is constructed and then processed by the decoder convolutional layers.

If you utilize any portion or the entirety of this code, kindly cite the paper titled "SLO-MSNet: Discrimination of Multiple Sclerosis using Scanning Laser Ophthalmoscopy Images" DOI: https://doi.org/10.1101/2023.09.03.23294985 
 

First create a dictionary for your dataset in which each key refers to one patient and the value of each key must be a nested-dictionary with own key and value. Keyes in nested dictionary indicate the number of images belonging to the patient and values are the corresponding numpy array of the images. Save this dictionary as a pickle file named (“subjects_slo_data.pkl”). However, the label dictionary contains keys and values, where the keys are the same as the keys in the image dictionary and the values are the patient’s label. Save this dictionary as a pickle file named (“labels_slo_data.pkl”)

For example patient number one has 4 images with label = 1 and patient number two has 2 images with label = 0. Therefore, the corresponding dictionaries are as follow:

•   images [0] is a dictionary with size (4):

•   np.shape(images[0][0])  = (128 ×128 ×1)

•   np.shape(images [0][1]) = (128 ×128 ×1)

•   np.shape(images [0][2]) = (128 ×128 ×1)

•   np.shape(images [0][3]) = (128 ×128 ×1)

 

   labels_train [0] = 1

 

•   images [1] is a dictionary with size (2):

•   np.shape(images [1][0]) = (128 ×128 ×1)

•   np.shape(images [1][1]) = (128 ×128 ×1)

 

   labels_train [1] = 0

 

Remember to resize your images to a square size like (128 × 128 × 1).

Then in order to extract the features using the proposed Conventional AutoEncoder (CAE). Run the file “feature_extaction.py”

By running the mentioned code, the extracted features are saved for further processing.

To complete the classification with the proposed method named “SLO-MSNet” run the code “mlp_classifiaction”.

 

"""
