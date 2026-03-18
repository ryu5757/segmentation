# プロジェクト名：Segmentation

## 1. 実行環境の準備
必要なライブラリをインポートし、GPUの設定を行います。

```python
import os

# メモリ管理の設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
