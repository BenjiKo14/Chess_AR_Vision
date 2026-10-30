import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

def train_mobilenet(dataset_dir="dataset", output_onnx="models/chess_mobilenetv3.onnx", epochs=15, batch_size=32, lr=1e-3):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision import datasets, transforms, models
    except ImportError:
        print("❌ PyTorch ou Torchvision manquant. Installez-les avec: pip install torch torchvision")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Début de l'entraînement sur : {device}")

    Path(output_onnx).parent.mkdir(parents=True, exist_ok=True)

    data_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms)
    if len(dataset) < 13:
        print("❌ Dataset insuffisant (< 13 images). Capturez d'abord quelques positions.")
        return False

    # Calcul des poids de classes pour équilibrer pions / dames / rois
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=len(dataset.classes))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += imgs.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total:.4f} - Acc: {acc*100:.1f}%")

    # Export universel ONNX (compatible OpenCV DNN en 100% CPU sans PyTorch)
    print(f"💾 Export ONNX vers {output_onnx}...")
    model.eval()
    dummy_input = torch.randn(64, 3, 96, 96, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    meta_file = Path(output_onnx).parent / "model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "classes": dataset.classes,
            "epochs": epochs,
            "accuracy": round(acc, 4)
        }, f, indent=2)

    print("✅ Modèle ONNX prêt pour inférence CPU !")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Dossier du dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'epochs")
    args = parser.parse_args()
    train_mobilenet(dataset_dir=args.dataset, epochs=args.epochs)
