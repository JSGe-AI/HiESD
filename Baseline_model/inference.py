
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import os
from tqdm import tqdm
import argparse # 导入 argparse

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class H5Dataset(Dataset):
    def __init__(self, folder_path, target_labels):
        self.data = []
        self.target_labels = target_labels
        h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        if not h5_files:
            print(f"Warning: No .h5 files found in {folder_path}")
            # You might want to raise an error or handle this case appropriately
            # raise FileNotFoundError(f"No .h5 files found in {folder_path}")

        for file_path in h5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'labels' not in f or 'features' not in f:
                        print(f"Warning: 'labels' or 'features' dataset not found in {file_path}. Skipping.")
                        continue
                    labels = f['labels'][...]
                    features = f['features'][...]
                    mask = np.isin(labels, self.target_labels)
                    self.data.extend(list(zip(features[mask], labels[mask])))
            except Exception as e:
                print(f"Error reading {file_path}: {e}. Skipping.")
        
        if not self.data:
            print(f"Warning: No data loaded for target labels {target_labels} from {folder_path}.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        # Remap labels to 0, 1, ..., num_classes-1 if necessary for CrossEntropyLoss
        # This assumes target_labels are sorted or can be mapped to indices
        mapped_label = self.target_labels.index(label) if label in self.target_labels else -1 # Should always be in target_labels due to mask
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(mapped_label, dtype=torch.long)


def main(args):

    model_name = args.model_name
    input_size = args.input_size
    hidden_size = args.hidden_size

    batch_size = args.batch_size

    target_labels = args.target_labels
    num_classes = len(target_labels)
    folds_to_process = args.folds

    base_val_dir_template = args.val_dir_template # "/data_nas2/gjs/ESD_2025/test/classfication/XJ/"
    base_ckpt_dir = args.ckpt_dir # "/home/gjs/ESD_2025/Experiment/ckpt/"
    base_results_dir = args.results_dir # "/home/gjs/ESD_2025/Experiment/results/"

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    for fold_num in folds_to_process:
        print(f"--- {model_name} inference fold {fold_num} ---")

        # val_dir = f"/data_nas2/gjs/ESD_2025/test/classfication/XJ/{model_name}_{num_classes}cls/feature_label"
        val_dir = os.path.join(base_val_dir_template, f"{model_name}_{num_classes}cls", "feature_label")
        print(f"Validation data directory: {val_dir}")
        if not os.path.isdir(val_dir):
            print(f"Error: Validation directory {val_dir} not found for fold {fold_num}. Skipping.")
            continue
            
        val_dataset = H5Dataset(val_dir, target_labels)
        if len(val_dataset) == 0:
            print(f"Warning: Validation dataset for fold {fold_num} is empty. Skipping.")
            continue
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        model = MLP(input_size, hidden_size, num_classes).to(device)
        
        # model_path = f"/home/gjs/ESD_2025/Experiment/ckpt/{model_name}/{num_classes}cls_best_fold_{fold_num}.pth"
        model_path = os.path.join(base_ckpt_dir, model_name, f"{num_classes}cls_best_fold_{fold_num}.pth")
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint {model_path} not found for fold {fold_num}. Skipping.")
            continue
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model state_dict from {model_path}: {e}. Skipping fold {fold_num}.")
            continue
            
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer not really needed for eval

        best_accuracy = 0.0 # This seems to be for a single epoch, which is fine for inference

        for epoch in range(1): # Usually just 1 epoch for inference
            model.eval()
            with torch.no_grad():
                all_labels = []
                all_predicted = []
                loop_val = tqdm(enumerate(val_loader), total=len(val_loader), leave=True, desc=f"Fold {fold_num} Val")
                
                for i, (features, labels) in loop_val:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())
                    
                    loop_val.set_description(f"Fold {fold_num}")
                    loop_val.set_postfix(val_loss=val_loss.item())

                if not all_labels: # Check if any data was processed
                    print(f"Warning: No labels collected for fold {fold_num}, epoch {epoch+1}. Skipping metrics calculation.")
                    continue

                # Remap predicted labels back to original values if needed for classification_report
                # The H5Dataset now maps labels to 0..N-1 for the loss function.
                # For reporting, we should use these 0..N-1 indices if target_names are also 0..N-1.
                # Or, if target_names use original labels, map predicted back.
                # Current approach: H5Dataset maps to 0..N-1, so use 0..N-1 for report labels.
                report_labels = list(range(num_classes))
                report_target_names = [str(tl) for tl in target_labels] # Use original labels for target_names

                overall_accuracy = accuracy_score(all_labels, all_predicted)
                # recall = recall_score(all_labels, all_predicted, average=None, labels=report_labels, zero_division=0)
                # f1 = f1_score(all_labels, all_predicted, average=None, labels=report_labels, zero_division=0)
                report = classification_report(all_labels, all_predicted, labels=report_labels, target_names=report_target_names, output_dict=True, zero_division=0)

                print(f'Fold {fold_num} , Overall Accuracy: {100 * overall_accuracy:.2f}%')
                # print(f'Per-class recall: {recall}')
                # print(f'Per-class F1-score: {f1}')

                if overall_accuracy > best_accuracy: # For inference (1 epoch), this will always be true on the first run
                    best_accuracy = overall_accuracy
                    
                    # results_file_path = f"/home/gjs/ESD_2025/Experiment/results/{model_name}/XJ_{num_classes}cls_fold_{fold_num}.txt"
                    results_dir_for_model = os.path.join(base_results_dir, model_name)
                    os.makedirs(results_dir_for_model, exist_ok=True) 
                    results_file_path = os.path.join(results_dir_for_model, f"XJ_{num_classes}cls_fold_{fold_num}.txt")
                    
                    print(f"Saving results to: {results_file_path}")
                    with open(results_file_path, "w") as f_out:
                        f_out.write(f"Fold {fold_num} - Overall Accuracy: {100 * overall_accuracy:.2f}%\n\n")
                        f_out.write(classification_report(all_labels, all_predicted, labels=report_labels, target_names=report_target_names, zero_division=0))
                        f_out.write("\n\nDetailed per-class metrics (from dict):\n")
                        for original_label_str in report_target_names: # Iterate using original label strings as keys in report
                            if original_label_str in report:
                                metrics = report[original_label_str]
                                f_out.write(f"Class {original_label_str}:\n")
                                f_out.write(f"  Precision: {metrics['precision']:.2f}\n")
                                f_out.write(f"  Recall: {metrics['recall']:.2f}\n")
                                f_out.write(f"  F1-score: {metrics['f1-score']:.2f}\n")
                                f_out.write(f"  Support: {metrics['support']}\n\n")
                            else:
                                # This case might happen if a class had zero true samples AND zero predicted samples
                                # depending on how classification_report handles it.
                                print(f"Warning: Metrics for class {original_label_str} not found in report dict for fold {fold_num}.")
                    
                    print("Detailed classification report:")
                    for original_label_str in report_target_names:
                        if original_label_str in report:
                            metrics = report[original_label_str]
                            print(f"Class {original_label_str}:")
                            print(f"  Precision: {metrics['precision']:.2f}")
                            print(f"  Recall: {metrics['recall']:.2f}")
                            print(f"  F1-score: {metrics['f1-score']:.2f}")
                            print(f"  Support: {metrics['support']}")
        print(f"--- Finished fold {fold_num} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP Model Inference Script")
    

    parser.add_argument('--model-name', type=str, default='UNI', help='Name of the model')
    parser.add_argument('--input-size', type=int, default=1024, help='Input size for MLP')
    parser.add_argument('--hidden-size', type=int, default=2048, help='Hidden size for MLP')
    
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for DataLoader')

    

    parser.add_argument('--target-labels', type=int, nargs='+', default=[0, 1], help='List of target labels to filter and classify')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 6)), help='List of fold numbers to process (e.g., 1 2 3 4 5)')
    

    parser.add_argument('--val-dir-template', type=str, 
                        default="/data_nas2/gjs/ESD_2025/test/classfication/XJ/",
                        help='Base template path for validation data. Structure: {val_dir_template}/{model_name}_{num_classes}cls/feature_label')
    parser.add_argument('--ckpt-dir', type=str, 
                        default="/home/gjs/ESD_2025/Experiment/ckpt/",
                        help='Base directory where model checkpoints are stored. Structure: {ckpt_dir}/{model_name}/{num_classes}cls_best_fold_{fold_num}.pth')
    parser.add_argument('--results-dir', type=str, 
                        default="/home/gjs/ESD_2025/Experiment/results/",
                        help='Base directory to save result files. Structure: {results_dir}/{model_name}/XJ_{num_classes}cls_fold_{fold_num}.txt')
    
   
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation (cuda or cpu)')

    args = parser.parse_args()
    
 
    if not isinstance(args.target_labels, list):
        args.target_labels = [args.target_labels]

    args.target_labels.sort()


  
    if not isinstance(args.folds, list):
        args.folds = [args.folds]

    print("Running with the following configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 30)

    main(args)

        