# **README**

This code repository provides data preprocessing code for the ESD dataset and training code for several baseline models.

## **Data preprocessing**

The image patches can be obtained using the general WSIs preprocessing framework. 

Here we provide a way to get image patches. Execute the following command :

```
python WSI_process.py \
    --svs_dir "path/to/your/new_data/svs" \
    --mask_components_dir "path/to/your/new_data/ESD_components_104" \
    --summary_dir "path/to/your/new_data/ESD_mask" \
    --output_dir "path/to/your/desired/final_data" \
    --patch_size 1024 \
    --white_ratio_threshold 0.8
```

## **Baseline Model**

The trained weights for the classification/segmentation models can be downloaded at this link:https://huggingface.co/datasets/JSGe-AI/HiESD/tree/main/Checkpoints



### Classification

The classification task uses ResNet-50, CONCH and UNI as baseline models, and the patch size is 1024×1024 at 40x.

#### ResNet-50

For specific code, see  *Baseline_model/ResNet.py*. When training on the ESD dataset, pre-load the weights pre-trained on ImageNet.

#### CONCH & UNI

Use CONCH and UNI official codes to pre-extract features from image patches. The code link can be found in 

https://github.com/mahmoodlab/CONCH

https://github.com/mahmoodlab/UNI

Execute the following command for training. The `--data_root` contains the 5-fold cross validation data, and each subfolder such as  `fold_1/train`  and   `fold_1/val`  contains the `.h5`  feature files.  The `--result_dir` argument defines the directory where evaluation results (such as accuracy, precision, recall, and F1 scores) will be saved. The `--ckpt_dir` argument sets the directory for saving the best-performing model checkpoints during training. These parameters allow flexible configuration of data access and output management for experiments. 

```
python conch_train.py \
  --data_root /data_nas2/gjs/ESD_2025/classification/CONCH_4class/5_fold \
  --result_dir /home/gjs/ESD_2025/Experiment/results/CONCH \
  --ckpt_dir /home/gjs/ESD_2025/Experiment/ckpt/CONCH \
  --input_size 512 \
  --hidden_size 2048 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --num_epochs 50 \
  --target_labels 0,1,2,3

```

```
python UNI_train.py \
  --data_root /data_nas2/gjs/ESD_2025/classification/UNI_4cls/5_fold \
  --result_dir /home/gjs/ESD_2025/Experiment/results/UNI \
  --ckpt_dir /home/gjs/ESD_2025/Experiment/ckpt/UNI \
  --exp_name 4cls \
  --input_size 1024 \
  --hidden_size 2048 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --num_epochs 50 \
  --target_labels 0,1,2,3
```

### segmentation

See *Baseline_model/ResUnet_train.py* for the segmentation code, using ResNet-50 as the backbone.

**'--train_txt_path'** indicates the txt file where the training set image path is stored.

**'--val_txt_path'** indicates the path of the txt file storing the validation set images.

**'--mask_replace_from'** and **'--mask_replace_to'** means switching from image path to the corresponding mask path. (e.g. from '.../train/image/001.png' to '.../train/mask/001.png' )

```
python ResUnet_train.py \
    --train_txt_path "/home/ESD_2025/Segment/5fold_data/fold_1_train.txt" \
    --val_txt_path "/home/ESD_2025/Segment/5fold_data/fold_1_val.txt" \
    --output_dir "/home/ESD_2025/Segment/run_outputs" \
    --num_classes 5 \
    --fold 1 \
    --epochs 60 \
    --batch_size 16 \
    --lr 0.001 \
    --encoder_name "resnet50" \
    --encoder_weights "imagenet" \
    --mask_replace_from "image" \
    --mask_replace_to "mask" \
    --suffix "experiment_A"
```

## Inference

To perform inference on the test set using the trained weights, use the following code command:

```
python inference.py --model-name UNI --input-size 1024
```

## Visualization

To get a visual map of the prediction results, execute the following command:

```
python Visualization.py \
    --num_classes 5 \
    --model_name "MyModel" \
    --ckpt_base_dir "/path/to/your/checkpoints/" \
    --data_root_base_dir "/path/to/your/h5_data/" \
    --output_dir "./mymodel_predictions_5cls" \
    --device "cuda"
```

