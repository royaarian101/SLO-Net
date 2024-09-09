

"""

***SLO-Net:***


In this study, we introduce SLO-Net, a novel bi-modal model designed for diagnosing multiple sclerosis (MS). This approach utilizes a dataset from Isfahan, Iran, which includes IR-SLO images and OCT data from 32 MS patients and 70 healthy individuals.

We trained several convolutional neural networks (CNNs)—namely, VGG-16, VGG-19, ResNet-50, ResNet-101, and a custom architecture—using both IR-SLO images and OCT thickness maps as separate input modalities. The best-performing models for each modality (ResNet-101 for both) were then combined to create a bi-modal model that integrates both OCT thickness maps and IR-SLO images.

Incorporating both modalities enhances the performance of automated MS diagnosis, highlighting the value of using IR-SLO as a complementary tool alongside OCT.


**To run this code, you first need to create properly formatted training and testing dataset dictionaries. Below, we discuss our dataset and the process for creating the input data. You can follow a similar approach to prepare your own dataset.**

Our dataset consists of IR-SLO and OCT scans from 102 individuals, including 32 with multiple sclerosis (MS) and 70 healthy controls (HC).


*  *IR-SLO images:*
IR-SLO images were gray-scale images of size 768 × 768 × 1, with all being finally resized to 128 × 128 × 1. 
Note that all IR-SLO images of left eyes were mirrored to ensure uniform orientation across all images.



*  *OCT scans:*
OCT B-scans were initially segmented into nine layers using previously developed code (1). For this study, we included only the thickness maps of the total retina. Therefore, the OCT thickness maps were generated by subtracting the first boundary from the last. Finally, each thickness map was resized to a 60 × 256 × 1 image format. 
Please be aware that OCT images of left eyes were mirrored as well. 


*  *Final merged dataset:*
The training and internal test datasets were separately saved into two Python dictionaries, named "train_merged_data_with_sp" and "test_merged_data_with_sp", respectively. Each dictionary uses keys to represent the indices assigned to each subject, ranging from key “0” to key “101”.


(Note that there was no key overlap between the two datasets. This was a result of our dedication to a subject-wise approach, in which we initially employed random sampling and assigned 21 subjects (20% of the total) to the internal test dataset, leaving the others for the training dataset.)


The value corresponding to each key was made up of a list containing pairs of IR-SLO and OCT thickness maps for each subject. Each subject could have one or multiple pairs. 


In addition to the dictionaries for saving the merged image datasets (IR-SLO image and OCT thickness map pairs), two additional dictionaries were created for storing labels: one for training, named "train_merged_data_with_sp_labels", and one for internal testing, named "test_merged_data_with_sp_labels". These dictionaries use subject indices as keys, with values representing the subject's label (0 for healthy controls and 1 for multiple sclerosis).


All the dictionaries were saved in pickle file format for further processing.


**Now run the file “Bimodal_model_classifier.py” with your prepared datasets.**

The file "merged_model.ipynb" contains the equivalent code for execution in Google Colab.


*For the **Optuna** code, please refer to : https://github.com/royaarian101/Optuna*




reference: 
(1) Kafieh R, Rabbani H, Abramoff MD, Sonka M. Intra-retinal layer segmentation of 3D optical coherence tomography using coarse grained diffusion map. Medical Image Analysis. 2013 Dec;17(8):907–28. 



**Please ensure to include the following citations when utilizing any part of the code:**

**[1] Arian, R., Aghababaei, A., Soltanipour, A., Khodabandeh, Z., Rakhshani, S., Iyer, S. B., Ashtari, F., Rabbani, H., & Kafieh, R. (2024). SLO-net: Enhancing multiple sclerosis diagnosis beyond optical coherence tomography using infrared reflectance scanning laser ophthalmoscopy images. Translational Vision Science & Technology, 13(7), 13. https://doi.org/10.1167/tvst.13.7.13**

**[2] Aghababaei A, Arian R, Soltanipour A, Ashtari F, Rabbani H, Kafieh R. Discrimination of Multiple Sclerosis using Scanning Laser Ophthalmoscopy Images with Autoencoder-Based Feature Extraction. Multiple Sclerosis and Related Disorders. 2024 Aug 1;88:105743–3.**

"""
