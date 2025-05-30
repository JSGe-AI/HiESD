# **README**

This code repository provides data preprocessing code for the ESD dataset and training code for several baseline models.

## **Data preprocessing**

The image patches can be obtained using the general WSIs preprocessing framework. 

Here we provide a way to get image patches. Execute the following command :

```
python data_process/WSI_process.py
```

## **Baseline Model**

The trained weights for the classification/segmentation models can be downloaded at this link:https://huggingface.co/datasets/JSGe-AI/HiESD/tree/main/Checkpoints



### Classification

The classification task uses ResNet-50, CONCH and UNI as baseline models, and the patch size is 1024×1024 at 40x.

#### ResNet-50

For specific code, see  *Baseline_model/ResNet.py*. When training on the ESD dataset, pre-load the weights pre-trained on ImageNet.

#### CONCH

Use CONCH's official code to extract features from image patches in advance. The code link can be found in 

https://github.com/mahmoodlab/CONCH

Execute the following command for training: 

```
python conch_train.py
```

#### UNI

Use UNI's official code to extract features from image patches in advance. The code link can be found in 

https://github.com/mahmoodlab/UNI

Execute the following command for training: 

```
python UNI_train.py
```

### segmentation

See *Baseline_model/ResUnet_train.py* for the segmentation code, using ResNet-50 as the backbone.

```
python ResUnet_train.py
```

## Inference

To perform inference on the test set using the trained weights, use the following code command:

```
python inference.py
```

## Visualization

To get a visual map of the prediction results, execute the following command:

```
python Visualization.py
```

