import os
import torch
import timm
import torchmetrics
from images_dataset import ImageDataset, class_mapping
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm


def main():
    root = 'hollywood_data'

    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    images = ImageDataset(root_dir=root, transform=transform)
    num_images = len(images)
    train_num = int(num_images * 0.9)
    val_num = int(num_images * 0.05)
    test_num = num_images - train_num - val_num

    batch_size = 32
    num_workers = 4
    # lr = 3e-4
    epochs = 10

    train_data, val_data, test_data = torch.utils.data.random_split(images, [train_num, val_num, test_num])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    classes = sorted(class_mapping.keys())

    rexnet_model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(classes))

    def to_device(batch, device_type):
        return batch[0].to(device_type), batch[1].to(device_type)

    def train_setup(model, num_epochs, lr=3e-4):
        return model.to("cuda").train(), num_epochs, "cuda", CrossEntropyLoss(), Adam(params=model.parameters(), lr=lr)

    def get_metrics(model, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1, f1_score):
        preds = model(ims)
        loss = loss_fn(preds, gts)
        epoch_loss += loss.item()
        epoch_acc += (torch.argmax(preds, dim=1) == gts).sum().item()
        epoch_f1 += f1_score(preds, gts).item()
        return loss, epoch_loss, epoch_acc, epoch_f1

    rexnet_model, epochs, device, loss_fn, optimizer = train_setup(rexnet_model, epochs)

    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)
    save_prefix, save_dir = "faces", "saved_models"
    print("Start training...")

    best_loss = float("inf")
    threshold, not_improved, patience = 0.01, 0, 5
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = [], [], [], [], [], []

    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0
        rexnet_model.train()
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            ims, gts = to_device(batch, device)
            optimizer.zero_grad()

            loss, epoch_loss, epoch_acc, epoch_f1 = get_metrics(
                rexnet_model, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1, f1_score
            )
            loss.backward()
            optimizer.step()

        tr_loss_to_track = epoch_loss / len(train_loader)
        tr_acc_to_track = epoch_acc / len(train_loader.dataset)
        tr_f1_to_track = epoch_f1 / len(train_loader)
        train_losses.append(tr_loss_to_track)
        train_accs.append(tr_acc_to_track)
        train_f1s.append(tr_f1_to_track)

        print(
            f"Epoch {epoch + 1} Training Loss: {tr_loss_to_track:.3f}, Accuracy: {tr_acc_to_track:.3f}, F1: {tr_f1_to_track:.3f}")

        # Validation phase
        rexnet_model.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc, val_epoch_f1 = 0, 0, 0
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                ims, gts = to_device(batch, device)
                loss, val_epoch_loss, val_epoch_acc, val_epoch_f1 = get_metrics(
                    rexnet_model, ims, gts, loss_fn, val_epoch_loss, val_epoch_acc, val_epoch_f1, f1_score
                )

            val_loss_to_track = val_epoch_loss / len(val_loader)
            val_acc_to_track = val_epoch_acc / len(val_loader.dataset)
            val_f1_to_track = val_epoch_f1 / len(val_loader)
            val_losses.append(val_loss_to_track);
            val_accs.append(val_acc_to_track);
            val_f1s.append(val_f1_to_track)

            print(
                f"Epoch {epoch + 1} Validation Loss: {val_loss_to_track:.3f}, Accuracy: {val_acc_to_track:.3f}, F1: {val_f1_to_track:.3f}")

            # Model checkpoint
            if val_loss_to_track < (best_loss - threshold):
                os.makedirs(save_dir, exist_ok=True)
                best_loss = val_loss_to_track
                torch.save(rexnet_model.state_dict(), f"{save_dir}/{save_prefix}_best_model.pth")
            else:
                not_improved += 1
                print(f"Loss did not improve for {not_improved} epochs")
                if not_improved == patience:
                    print(f"Stopping training after {patience} epochs without improvement.")
                    break


if __name__ == '__main__':
    main()
