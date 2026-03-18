```python
# ============================================================
# 1. Library Imports
# ============================================================

# 標準ライブラリ
import os
import time
import random
from dataclasses import dataclass

# 環境変数の設定 (GPUメモリ管理)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# サードパーティライブラリ
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat

# Optuna (ハイパーパラメータ最適化)
# !pip install optuna  # 必要に応じて実行環境で実行してください
import optuna

# PyTorch 関連
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
from torch.cuda.amp import autocast, GradScaler

# torchvision 関連
from torchvision.datasets import VisionDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    GaussianBlur,
    Resize,
    ToTensor,
    Normalize,
    Lambda,
    InterpolationMode
)
```
# DataLoader
# カラーマップ生成関数：セグメンテーションの可視化用
def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# NYUv2データセット：RGB画像、セグメンテーション、深度、法線マップを提供するデータセット
class NYUv2(VisionDataset):
    """NYUv2 dataset

    Args:
        root (string): Root directory path.
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, ``semantic``, ``depth``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    cmap = colormap()
    def __init__(self,
                 root,
                 split='train',
                 include_depth=False,
                 transform=None,
                 target_transform=None,
                 ):
        super(NYUv2, self).__init__(root, transform=transform, target_transform=target_transform)

        # データセットの基本設定
        assert(split in ('train', 'test'))
        self.root = root
        self.split = split
        self.include_depth = include_depth
        self.train_idx = np.array([255, ] + list(range(13)))  # 13クラス分類用

        # 画像ファイルのパスリストを作成
        img_names = os.listdir(os.path.join(self.root, self.split, 'image'))
        img_names.sort()
        images_dir = os.path.join(self.root, self.split, 'image')
        self.images = [os.path.join(images_dir, name) for name in img_names]

        label_dir = os.path.join(self.root, self.split, 'label')
        if (self.split == 'train'):
          self.labels = [os.path.join(label_dir, name) for name in img_names]
          self.targets = self.labels

        depth_dir = os.path.join(self.root, self.split, 'depth')
        self.depths = [os.path.join(depth_dir, name) for name in img_names]

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        depth = Image.open(self.depths[idx])

        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)
        if self.split=='test':
          if self.include_depth:
              return image, depth
          return image
        if self.split == 'train' and self.target_transform is not None:
            target = Image.open(self.targets[idx])
            target = self.target_transform(target)
        if self.include_depth:
              return image, depth, target

        return image, target

    def __len__(self):
        return len(self.images)
# Model Section
# 2つの畳み込み層とバッチ正規化、ReLUを含むブロック
# UNetの各層で使用される基本的な畳み込みブロック
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# UNetモデル：エンコーダ・デコーダ構造のセグメンテーションモデル


from torchvision import models

class CustomUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        features = base_model.features


        first_conv = features[0][0]
        new_conv = nn.Conv2d(in_channels, first_conv.out_channels,
                             kernel_size=first_conv.kernel_size, stride=first_conv.stride,
                             padding=first_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            new_conv.weight[:, 3:] = first_conv.weight.mean(dim=1, keepdim=True)
        features[0][0] = new_conv



        self.b0_1 = nn.Sequential(features[0], features[1]) # 1/2
        self.b2   = features[2] # 1/4
        self.b3   = features[3] # 1/8
        self.b4_5 = nn.Sequential(features[4], features[5]) # 1/16
        self.b6   = features[6]
        self.b7   = features[7] # 1/32, 384ch


        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


        self.dec4 = DoubleConv(384 + 136, 256)
        self.dec3 = DoubleConv(256 + 48, 128)
        self.dec2 = DoubleConv(128 + 32, 64)
        self.dec1 = DoubleConv(64 + 24, 32)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.b0_1(x)    # 1/2
        e2 = self.b2(e1)     # 1/4
        e3 = self.b3(e2)     # 1/8
        e4 = self.b4_5(e3)   # 1/16


        f6 = self.b6(e4)
        e5 = self.b7(f6)    # 1/32

        # e4 と up_e5 のサイズを比較して、足りない分をパディング
        up_e5 = self.up(e5)
        diff_h4 = e4.size()[2] - up_e5.size()[2]
        diff_w4 = e4.size()[3] - up_e5.size()[3]
        up_e5_pad = torch.nn.functional.pad(up_e5, [diff_w4 // 2, diff_w4 - diff_w4 // 2,
                                               diff_h4 // 2, diff_h4 - diff_h4 // 2])
        d4 = self.dec4(torch.cat([up_e5_pad, e4], dim=1))

        up_d4 = self.up(d4)
        diff_h3 = e3.size()[2] - up_d4.size()[2]
        diff_w3 = e3.size()[3] - up_d4.size()[3]
        up_e4_pad = torch.nn.functional.pad(up_d4, [diff_w3 // 2, diff_w3 - diff_w3 // 2,
                                               diff_h3 // 2, diff_h3 - diff_h3 // 2])
        d3 = self.dec3(torch.cat([up_e4_pad, e3], dim=1))

        up_d3 = self.up(d3)
        diff_h2 = e2.size()[2] - up_d3.size()[2]
        diff_w2 = e2.size()[3] - up_d3.size()[3]
        up_d3_pad = torch.nn.functional.pad(up_d3, [diff_w2 // 2, diff_w2 - diff_w2 // 2,
                                               diff_h2 // 2, diff_h2 - diff_h2 // 2])
        d2 = self.dec2(torch.cat([up_d3_pad, e2], dim=1))

        up_d2 = self.up(d2)
        diff_h1 = e1.size()[2] - up_d2.size()[2]
        diff_w1 = e1.size()[3] - up_d2.size()[3]
        up_d2_pad = torch.nn.functional.pad(up_d2, [diff_w1 // 2, diff_w1 - diff_w1 // 2,
                                               diff_h1 // 2, diff_h1 - diff_h1 // 2])
        d1 = self.dec1(torch.cat([up_d2_pad, e1], dim=1))



        return self.final(self.final_up(d1))
# Train and Valid
# config
@dataclass
class TrainingConfig:
    # データセットパス
    dataset_root: str = "data"

    # データ関連
    batch_size: int = 32
    num_workers: int = 4# RGB + Depth

    # モデル関連
    in_channels: int = 4 # RGB + Depth
    num_classes: int = 13  # NYUv2データセットの場合

    # 学習関連
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # データ分割関連
    train_val_split: float = 0.8  # 訓練データの割合

    # デバイス設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # チェックポイント関連
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5  # エポックごとのモデル保存間隔

    # データ拡張・前処理関連
    image_size: tuple = (256, 256)
    normalize_mean: tuple = (0.485, 0.456, 0.406)  # ImageNetの標準化パラメータ
    normalize_std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

def set_seed(seed):
    """
    シードを固定する．

    Parameters
    ----------
    seed : int
        乱数生成に用いるシード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# 設定の初期化
config = TrainingConfig(
    dataset_root='/content/data',
    batch_size=16,
    num_workers=2,
    learning_rate=1e-4,
    weight_decay = 1e-4,
    epochs=100,
    image_size=(320, 240),
    in_channels=4 ) # RGB(3チャネル) + Depth(1チャネル)



'''
データセットのディレクトリ構造：
    data/NYUv2/
    ├─train/
    │  ├─image/      # RGB画像（入力）
    │  │    000000.png
    │  │    ...
    |  ├─depth/      # 深度画像（入力）
    |  │    000000.png
    |  │    ...
    │  └─label/      # 13クラスセグメンテーション（教師ラベル）
    │       000000.png
    │       ...
    └─test/
       ├─image/      # RGB画像（入力）
       │    000000.png
       │    ...
       ├─depth/      # 深度画像（入力）
       │    000000.png
       │    ...
'''


# ------------------
#    Dataloader
# ------------------

# データ前処理の定義
# RGB画像のTransform：
rgb_transform_final = Compose([
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 学習用 Depth変換
depth_transform_final = Compose([
    ToTensor()
])

# 学習用 Label変換
def label_transform_final(lbl):
    return torch.from_numpy(np.array(lbl)).long()

# --- テスト用 ---
test_rgb_transform = Compose([
    Resize(config.image_size, interpolation=InterpolationMode.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_depth_transform = Compose([
    Resize(config.image_size, interpolation=InterpolationMode.BILINEAR),
    ToTensor()
])

test_target_transform = Compose([
    Resize(config.image_size, interpolation=InterpolationMode.NEAREST),
    Lambda(lambda lbl: torch.from_numpy(np.array(lbl)).long())
])

# テスト用 SmartTransform (RGBとDepthで処理を分ける)
class SmartTransform:
    def __init__(self, rgb_trans, depth_trans):
        self.rgb_trans = rgb_trans
        self.depth_trans = depth_trans

    def __call__(self, img):
        if img.mode == 'RGB':
            return self.rgb_trans(img)
        else:
            return self.depth_trans(img)

test_smart_transform = SmartTransform(test_rgb_transform, test_depth_transform)


class JointAugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_size, rgb_tf, depth_tf, label_tf):
        self.dataset = dataset
        self.image_size = image_size
        self.rgb_tf = rgb_tf
        self.depth_tf = depth_tf
        self.label_tf = label_tf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, depth, label = self.dataset[index]

        # 　ランダムスケール & クロップ
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(0.75, 1.333)
        )

        image = TF.resized_crop(image, i, j, h, w, self.image_size, InterpolationMode.BILINEAR)
        depth = TF.resized_crop(depth, i, j, h, w, self.image_size, InterpolationMode.NEAREST)
        label = TF.resized_crop(label, i, j, h, w, self.image_size, InterpolationMode.NEAREST)

        #  ランダム水平反転
        if random.random() > 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            label = TF.hflip(label)

        #  Small Rotation
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, InterpolationMode.BILINEAR)
        depth = TF.rotate(depth, angle, InterpolationMode.NEAREST)
        label = TF.rotate(label, angle, InterpolationMode.NEAREST)

        #  個別の変換
        image = self.rgb_tf(image)
        depth = self.depth_tf(depth)
        label = self.label_tf(label)

        return image, depth, label


identity_transform = Lambda(lambda x: x)


raw_train_dataset = NYUv2(
    root=config.dataset_root,
    split='train',
    include_depth=True,
    transform=None,
    target_transform=identity_transform
)

train_dataset = JointAugmentationDataset(
    raw_train_dataset,
    image_size=config.image_size,
    rgb_tf=rgb_transform_final,
    depth_tf=depth_transform_final,
    label_tf=label_transform_final
)

test_dataset = NYUv2(
    root=config.dataset_root,
    split='test',
    include_depth=True,
    transform=test_smart_transform,
    target_transform=test_target_transform
)



'''
    train data:
        Type of batch: tuple
        Index 0 (入力データ):
            Type: torch.Tensor
            Shape: torch.Size([Batch, 3, N, M])
            Details: RGBテンソル
                    - チャネル0-2: RGB画像 (値域: 0-1)
        Index 1 (教師ラベル):
            Type: torch.Tensor
            Shape: torch.Size([Batch, N, M])
            Details: セグメンテーションマップ
                    - 値域: 0-12 (13クラス)
                    - 255: ignore index

    test data:
        Type of batch: torch.Tensor
        Shape: torch.Size([Batch, 3, N, M])
        Details: RGB画像 (値域: 0-1)
'''

# データローダーの作成
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for image, depth, label in loader:
            # NYUv2クラスのtest実装に合わせて分岐
            if isinstance(image, tuple) or isinstance(image, list):
                 # もしgetitemが(image, depth)などを返す場合
                 image, depth = image
                 label = None

            image, depth, label = image.to(device), depth.to(device), label.to(device)

            with torch.cuda.amp.autocast():
                x = torch.cat((image, depth), dim=1)
                pred = model(x)
                loss = criterion(pred, label)

            total_loss += loss.item()

    return total_loss / len(loader)

# -------------------------------------------------------------
# 2. Optunaの目的関数
# -------------------------------------------------------------
def objective(trial):
    device = config.device
    # --- ハイパーパラメータの探索範囲定義 ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


    batch_size = 16

    # --- データセット準備 ---

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # num_workersは環境に合わせて調整
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- モデルセットアップ ---
    model = CustomUNet(in_channels=config.in_channels, num_classes=config.num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


    search_epochs = #探索用のエポック数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=search_epochs)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.cuda.amp.GradScaler()

    # --- 学習ループ ---
    model.train()
    for epoch in range(search_epochs):

        # Train
        for image, depth, label in train_loader:
            image, depth, label = image.to(device), depth.to(device), label.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                x = torch.cat((image, depth), dim=1)
                pred = model(x)
                loss = criterion(pred, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Validation
        val_loss = validate(model, val_loader, criterion, device)

        # --- Optunaへ報告 & 枝刈り ---
        trial.report(val_loss, epoch)

        # 結果が悪ければここで強制終了（時間短縮）
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

# -------------------------------------------------------------
# 3. 実行
# -------------------------------------------------------------
if __name__ == "__main__":
    n_trials = 20

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3, # 最低3エポックは回す
            max_resource=12,
            reduction_factor=3
        )
    )

    print("Start optimization...")
    study.optimize(objective, n_trials=n_trials)

    print("\n------------------------------------------------")
    print("Best Params:", study.best_params)
    print("Best Value (Val Loss):", study.best_value)
    print("------------------------------------------------\n")

    best_params = study.best_params
    config.learning_rate = best_params['lr']
    config.weight_decay = best_params['weight_decay']

    print(f"Starting Final Training with LR: {config.learning_rate}, WD: {config.weight_decay}")

    train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

# モデルとトレーニングの設定
    device = config.device
    print(f"Using device: {device}")

# ------------------
#    Model
# ------------------
    model = CustomUNet(in_channels=config.in_channels, num_classes=config.num_classes).to(device)

# ------------------
#    optimizer
# ------------------　　AdamからAdamWに変更
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)#スケジューラの導入
    criterion = nn.CrossEntropyLoss(ignore_index=255)

# ------------------
#    Training
# ------------------
    num_epochs = config.epochs
    scaler = GradScaler()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"on epoch: {epoch+1}")
        with tqdm(train_data) as pbar:

            for batch_idx, (image, depth, label) in enumerate(pbar):
                image, depth, label = image.to(device), depth.to(device), label.to(device)
                optimizer.zero_grad()

                with autocast():
                  x = torch.cat((image, depth), dim=1) # RGB + Depth
                  pred = model(x)
                  loss = criterion(pred, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                del image, depth, label, pred, loss

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

    # モデルの保存
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"model_{current_time}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# ------------------

#    Evaluation
# ------------------

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 予測結果の生成
predictions = []

with torch.no_grad():
    print("Generating predictions...")
    for image, depth in tqdm(test_data):
        image, depth = image.to(device), depth.to(device)
        x = torch.cat((image, depth), dim=1)
        output = model(x)            # [Batch, num_classes, H, W]
        pred = output.argmax(dim=1)  # [Batch, H, W]
        predictions.append(pred.cpu())
predictions = torch.cat(predictions, dim=0)

predictions = predictions.cpu().numpy()
np.save('submission.npy', predictions)
print("Predictions saved to submission.npy")
