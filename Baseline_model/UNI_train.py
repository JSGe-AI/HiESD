import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

import os
from tqdm import tqdm


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
        for file in [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]:
            with h5py.File(file, 'r') as f:
                # print(f"Features shape: {f['features'].shape}")
                # print(f"Labels shape: {f['labels'].shape}")
                labels = f['labels'][...]
                features = f['features'][...]
                mask = np.isin(labels, self.target_labels)
                self.data.extend(list(zip(features[mask], labels[mask])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



input_size = 1024
hidden_size = 2048
learning_rate = 1e-4
batch_size = 64
num_epochs = 50
target_labels = [0,1,2,3]  #
num_classes = len(target_labels) #


for d in range(1,6):

    fold_num = d

    train_dir = f"/data_nas2/gjs/ESD_2025/classification/UNI_4cls/5_fold/fold_{fold_num}/train"
    val_dir = f"/data_nas2/gjs/ESD_2025/classification/UNI_4cls/5_fold/fold_{fold_num}/val"



    train_dataset = H5Dataset(train_dir, target_labels)
    val_dataset = H5Dataset(val_dir, target_labels)


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)




 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize best accuracy
    best_accuracy = 0.0

   
    for epoch in range(num_epochs):
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)  
        
        for i, (features, labels) in loop:
            features = features.to(device)
            labels = labels.to(device)


       
            outputs = model(features)
            
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())  
                
        # Validation loop  (NEW CODE)
        model.eval()
        with torch.no_grad():
            all_labels = []
            all_predicted = []
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)  # leave=True
            
            for i, (features, labels) in loop_val:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy()) # Store all labels
                all_predicted.extend(predicted.cpu().numpy()) # Store all predictions
                
            
                loop_val.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                loop_val.set_postfix(val_loss=val_loss.item()) 

            # Calculate metrics
            overall_accuracy = accuracy_score(all_labels, all_predicted)
            recall = recall_score(all_labels, all_predicted, average=None) # per-class recall
            f1 = f1_score(all_labels, all_predicted, average=None) # per-class f1-score
            report = classification_report(all_labels, all_predicted, labels=target_labels, target_names=[str(x) for x in target_labels], output_dict=True, zero_division=1)


            print(f'Epoch [{epoch+1}/{num_epochs}], Overall Accuracy: {100 * overall_accuracy:.2f}%')
            print(f'Per-class recall: {recall}')
            print(f'Per-class F1-score: {f1}')


            # # Print detailed classification report with per-class precision, recall, f1-score
            # for label in target_labels:
            #     label_str = str(label)
            #     print(f"Class {label_str}:")
            #     print(f"  Precision: {report[label_str]['precision']:.2f}")
            #     print(f"  Recall: {report[label_str]['recall']:.2f}")
            #     print(f"  F1-score: {report[label_str]['f1-score']:.2f}")
            #     print(f"  Support: {report[label_str]['support']}") # Number of samples in this class
            
            # Save the best model based on overall accuracy
            if overall_accuracy > best_accuracy:
                
                # Print detailed classification report with per-class precision, recall, f1-score
                for label in target_labels:
                    label_str = str(label)
                    print(f"Class {label_str}:")
                    print(f"  Precision: {report[label_str]['precision']:.2f}")
                    print(f"  Recall: {report[label_str]['recall']:.2f}")
                    print(f"  F1-score: {report[label_str]['f1-score']:.2f}")
                    print(f"  Support: {report[label_str]['support']}") # Number of samples in this class
                    
                with open(f"/home/gjs/ESD_2025/Experiment/results/UNI/4cls_fold_{fold_num}.txt", "w") as f:
                    for label in target_labels:
                        label_str = str(label)
                        f.write(f"Class {label_str}:\n")
                        f.write(f"  Precision: {report[label_str]['precision']:.2f}\n")
                        f.write(f"  Recall: {report[label_str]['recall']:.2f}\n")
                        f.write(f"  F1-score: {report[label_str]['f1-score']:.2f}\n")
                        f.write(f"  Support: {report[label_str]['support']}\n\n")
                    
                
                best_accuracy = overall_accuracy
                print(f"New best accuracy: {100 * best_accuracy:.2f}%, saving model...")
                torch.save(model.state_dict(), f"/home/gjs/ESD_2025/Experiment/ckpt/UNI/4cls_best_fold_{fold_num}.pth") # Save model weights

        model.train() # Set the model back to training mode
        