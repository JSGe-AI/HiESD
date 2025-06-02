import argparse
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
                labels = f['labels'][...]
                features = f['features'][...]
                mask = np.isin(labels, self.target_labels)
                self.data.extend(list(zip(features[mask], labels[mask])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def main(args):
    target_labels = [int(x) for x in args.target_labels.split(',')]
    num_classes = len(target_labels)

    for d in range(1, 6):
        fold_num = d

        train_dir = os.path.join(args.data_root, f"fold_{fold_num}", "train")
        val_dir = os.path.join(args.data_root, f"fold_{fold_num}", "val")

        train_dataset = H5Dataset(train_dir, target_labels)
        val_dataset = H5Dataset(val_dir, target_labels)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLP(args.input_size, args.hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        best_accuracy = 0.0

        for epoch in range(args.num_epochs):
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            model.train()
            for i, (features, labels) in loop:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch [{epoch + 1}/{args.num_epochs}]")
                loop.set_postfix(loss=loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                all_labels = []
                all_predicted = []
                loop_val = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)

                for i, (features, labels) in loop_val:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())

                    loop_val.set_description(f"Epoch [{epoch + 1}/{args.num_epochs}]")
                    loop_val.set_postfix(val_loss=val_loss.item())

                overall_accuracy = accuracy_score(all_labels, all_predicted)
                recall = recall_score(all_labels, all_predicted, average=None)
                f1 = f1_score(all_labels, all_predicted, average=None)
                report = classification_report(all_labels, all_predicted,
                                               labels=target_labels,
                                               target_names=[str(x) for x in target_labels],
                                               output_dict=True, zero_division=1)

                print(f'Epoch [{epoch + 1}/{args.num_epochs}], Overall Accuracy: {100 * overall_accuracy:.2f}%')
                print(f'Per-class recall: {recall}')
                print(f'Per-class F1-score: {f1}')

                if overall_accuracy > best_accuracy:
                    for label in target_labels:
                        label_str = str(label)
                        print(f"Class {label_str}:")
                        print(f"  Precision: {report[label_str]['precision']:.2f}")
                        print(f"  Recall: {report[label_str]['recall']:.2f}")
                        print(f"  F1-score: {report[label_str]['f1-score']:.2f}")
                        print(f"  Support: {report[label_str]['support']}")

                    result_path = os.path.join(args.result_dir, f"4cls_conch_fold_{fold_num}.txt")
                    with open(result_path, "w") as f:
                        for label in target_labels:
                            label_str = str(label)
                            f.write(f"Class {label_str}:\n")
                            f.write(f"  Precision: {report[label_str]['precision']:.2f}\n")
                            f.write(f"  Recall: {report[label_str]['recall']:.2f}\n")
                            f.write(f"  F1-score: {report[label_str]['f1-score']:.2f}\n")
                            f.write(f"  Support: {report[label_str]['support']}\n\n")

                    best_accuracy = overall_accuracy
                    print(f"New best accuracy: {100 * best_accuracy:.2f}%, saving model...")
                    model_path = os.path.join(args.ckpt_dir, f"4cls_best_fold_{fold_num}.pth")
                    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP HDF5 classification")

    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--target_labels', type=str, default="0,1,2,3", help="Comma-separated class labels")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of 5-fold HDF5 data")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save result .txt files")
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Directory to save model checkpoints")

    args = parser.parse_args()
    main(args)
