# CSC4005 – Lab 2 Report: CNN vs Transfer Learning on NEU-CLS

## 1. Thông tin chung
- **Họ và tên**: Nguyễn Nam Cường
- **Lớp**: KHMT 16-01
- **Repo**: csc4005-lab2-cnn-neu-cuongnam
- **W&B project**: csc4005-lab2-neu-cnn
- **Dataset**: NEU-CLS (6 lớp: Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches)
- **Thời gian thực hiện**: May 6, 2026

## 2. Bài toán
Phân loại 6 lớp lỗi sản phẩm thép trên tập dữ liệu NEU-CLS. So sánh 2 cách tiếp cận:
1. **CNN from Scratch**: Xây dựng mô hình nhỏ (32K params) từ đầu
2. **Transfer Learning**: Sử dụng ResNet18 pre-trained trên ImageNet, chỉ fine-tune 3,078 params

Mục đích: Đánh giá hiệu suất, tốc độ huấn luyện, và khả năng generalization của hai phương pháp.

## 3. Mô hình và cấu hình

### 3.1. CNN from Scratch (Baseline 1)
**Cấu hình:**
```
Model Architecture: CNN-small (custom)
Train Mode: From scratch
Input Size: 64×64
Batch Size: 32
Optimizer: AdamW
Learning Rate: 0.001 (with decay: 0.0005 → 0.00025)
Weight Decay: 0.0001
Dropout: 0.3
Data Augmentation: Enabled
Epochs: 20
Early Stopping Patience: 5
Total Parameters: 32,614 (all trainable)
```

**Training Command:**
```bash
python -m src.train \
  --data_dir "D:\KHMT_16-01\Deep learning\NEU-CLS.zip" \
  --project csc4005-lab2-neu-cnn \
  --run_name cnn_small_baseline \
  --model_name cnn_small \
  --train_mode scratch \
  --optimizer adamw \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.3 \
  --epochs 20 \
  --batch_size 32 \
  --img_size 64 \
  --patience 5 \
  --augment \
  --use_wandb
```

### 3.2. Transfer Learning with ResNet18 (Best Model)
**Cấu hình:**
```
Model Architecture: ResNet18 (ImageNet pre-trained)
Train Mode: Transfer Learning (fine-tune only last layer)
Input Size: 128×128
Batch Size: 32
Optimizer: AdamW
Learning Rate: 0.001 (fixed, no decay)
Weight Decay: 0.0001
Dropout: 0.3
Data Augmentation: Enabled
Epochs: 10
Early Stopping Patience: 3
Total Parameters: 11,179,590
Trainable Parameters: 3,078 (99.97% frozen)
```

**Training Command:**
```bash
python -m src.train \
  --data_dir "D:\KHMT_16-01\Deep learning\NEU-CLS.zip" \
  --project csc4005-lab2-neu-cnn \
  --run_name resnet18_transfer \
  --model_name resnet18 \
  --train_mode transfer \
  --optimizer adamw \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.3 \
  --epochs 10 \
  --batch_size 32 \
  --img_size 128 \
  --patience 3 \
  --augment \
  --use_wandb
```

## 4. Bảng kết quả

| Metric | CNN Scratch | ResNet18 Transfer | Tốt hơn |
|--------|:----------:|:----------:|:-------:|
| **Best Val Accuracy** | 94.44% (epoch 16) | **96.67%** (epoch 10) | ResNet18 +2.23% |
| **Best Val Loss** | **0.1298** (epoch 16) | 0.1506 (epoch 10) | CNN -0.0208 |
| **Test Accuracy** | 94.81% | **96.30%** | ResNet18 +1.49% |
| **Test Loss** | 0.1375 | 0.1567 | CNN -0.0192 |
| **Avg Epoch Time** | **4.57s** | 20.65s | CNN 4.5x nhanh |
| **Total Params** | 32,614 | 11,179,590 | CNN nhẹ hơn |
| **Trainable Params** | 32,614 | 3,078 | ResNet18 ít hơn 90x |
| **Learning Curve** | ⚠️ Spike epoch 13 | **✅ Ổn định** | ResNet18 |
| **Overfitting** | Bất ổn (fluctuation) | **Ổn định** | ResNet18 |
| **Convergence** | ~5 epochs | **~2-3 epochs** | ResNet18 |

### 🏆 Best Model: **ResNet18 Transfer**
**Lý Do Chọn:**
1. ✅ Validation accuracy cao nhất (96.67%)
2. ✅ Learning curve ổn định (không spike)
3. ✅ Ít overfitting hơn (stable training)
4. ✅ Hội tụ nhanh (2-3 epochs)
5. ✅ Test generalization tốt (96.30% ≈ 96.67% val)

## 5. Phân tích Learning Curves

### 5.1. CNN from Scratch
```
📊 Train Loss:    1.399 → 0.173 (mượt mà)
📊 Val Loss:      1.748 → 0.134 (hội tụ ổn định)

⚠️ Vấn đề: Spike bất ngờ
   - Epoch 13: val_loss = 1.145 (tăng 8.6x) 💥
   - Epoch 15: val_loss = 0.267 (spike lại)
   - Epoch 17: val_loss = 0.427 (spike lần 3)
   → Learning curve không ổn định, nguy hiểm khi deploy

✅ Điểm tốt:
   - Hội tụ từ sớm (epoch 5: 94% val acc)
   - Train/val gap nhỏ (0.049)
```

### 5.2. ResNet18 Transfer Learning (Best)
```
📊 Train Loss:    1.347 → 0.227 (mượt mà, linear)
📊 Val Loss:      0.937 → 0.150 (mượt mà, linear)

✅ Điểm tốt:
   - Learning curve mượt mà, không spike
   - Hội tụ nhanh (epoch 2-3 cải thiện mạnh)
   - Đạt 96.67% val acc ở epoch 10
   - Learning curve dễ dự đoán

⚠️ Trade-off:
   - Train/val gap lớn hơn (0.076 vs 0.049)
   - Nhưng ổn định → không overfitting bất ổn
```

## 6. Classification Report & Confusion Matrix

### 6.1. ResNet18 Transfer (Test Set - 270 samples)

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support | Đánh Giá |
|-------|:---------:|:------:|:--------:|:-------:|---------|
| **Rolled-in_Scale** | 100% | 100% | **100%** ⭐ | 45 | Perfect |
| **Patches** | 95.74% | 100% | **97.83%** ✅ | 45 | Rất Tốt |
| **Crazing** | 100% | 95.56% | **97.73%** ✅ | 45 | Rất Tốt |
| **Scratches** | 95.65% | 97.78% | **96.70%** ✅ | 45 | Tốt |
| **Pitted_Surface** | 90.00% | 100% | **94.74%** | 45 | Tốt |
| **Inclusion** | 97.44% | 84.44% | **90.48%** ⚠️ | 45 | Yếu Nhất |
| **Macro Avg** | **96.47%** | **96.30%** | **96.24%** | 270 | - |
| **Weighted Avg** | **96.47%** | **96.30%** | **96.24%** | 270 | - |

**Phân Tích:**
- ✅ 3 lớp perfect: Rolled-in_Scale (100%), Patches (100%), Pitted_Surface (100%)
- ⚠️ Điểm yếu: Inclusion (90.48% F1, recall 84.44%)
- Sai lầm chính: Inclusion → Pitted_Surface (5 lỗi) → visual features tương tự

### 6.2. Confusion Matrix (ResNet18 Test Set)

```
                Dự Đoán
                CR  IN  PA  PS  RS  SC
Crazing         43  0   2   0   0   0     (2 sai)
Inclusion       0   38  0   5   0   2     (7 sai) ⚠️
Patches         0   0   45  0   0   0     (Perfect)
Pitted_Surface  0   0   0   45  0   0     (Perfect)
Rolled-in_Scale 0   0   0   0   45  0     (Perfect)
Scratches       0   1   0   0   0   44    (1 sai)

Tổng sai: 7/270 = 2.59%
```

**Sai Lầm Chi Tiết:**
1. Inclusion → Pitted_Surface (5 lỗi) - Khó phân biệt
2. Crazing → Patches (2 lỗi) - Visual tương tự
3. Inclusion → Scratches (2 lỗi) - Edge case
4. Scratches → Inclusion (1 lỗi) - Hiếm gặp

## 7. Kết luận

### 7.1. CNN có cải thiện so với việc không có?
✅ **Có, cải thiện đáng kể:**
- CNN from scratch: 94.81% test accuracy
- Phân loại tốt trên hầu hết các lớp
- Learning từ dữ liệu NEU-CLS cụ thể

### 7.2. Transfer Learning có tốt hơn CNN from Scratch không?
✅ **Có, tốt hơn trên hầu hết tiêu chí:**

| Tiêu Chí | CNN Scratch | ResNet18 | Tốt Hơn |
|---------|:----------:|:----------:|:-------:|
| **Test Acc** | 94.81% | **96.30%** ⬆️ | ResNet18 |
| **Val Acc** | 94.44% | **96.67%** ⬆️ | ResNet18 |
| **Learning Curve** | ⚠️ Spike | **✅ Ổn** | ResNet18 |
| **Generalization** | Bất ổn | **Ổn định** | ResNet18 |
| **Hội Tụ** | ~5 epochs | **~3 epochs** | ResNet18 |
| **Tốc Độ/epoch** | **4.57s** | 20.65s | CNN |

**Kết luận**: Transfer learning thắng 5/6 tiêu chí quan trọng. Chỉ CNN nhanh hơn nhưng điều đó không quan trọng bằng chất lượng.

### 7.3. Khi nào nên chọn Transfer Learning vs From Scratch?

#### ✅ Chọn Transfer Learning khi:
1. **Dữ liệu hạn chế** (< 10K images) → Tránh overfitting
2. **Cần accuracy cao** → Pre-trained features có sẵn
3. **Thời gian hạn chế** → Hội tụ nhanh (2-3 epochs)
4. **Lớp dữ liệu tương tự ImageNet** (vật thể, texture, etc.)
5. **Tài nguyên GPU có sẵn** → Chịu được inference chậm

**Ưu điểm:**
- ✅ Accuracy cao hơn (96.30% vs 94.81%)
- ✅ Learning curve ổn định
- ✅ Hội tụ nhanh
- ✅ Overfitting ít hơn
- ✅ Ít params trainable → Generalize tốt

#### ✅ Chọn From Scratch khi:
1. **Dữ liệu rất lớn** (> 100K images)
2. **Cần tốc độ inference nhanh** (embedded devices)
3. **Domain hoàn toàn khác ImageNet** (medical imaging, etc.)
4. **Tài nguyên GPU hạn chế** → Model nhỏ
5. **Linh hoạt tùy chỉnh architecture**

**Ưu điểm:**
- ✅ Model nhỏ (32K vs 11.2M params)
- ✅ Inference nhanh (4.57s vs 20.65s/epoch)
- ✅ Đơn giản, dễ debug

### 7.4. Khuyến nghị cho Lab này

🏆 **Best Model: ResNet18 Transfer Learning**
- Accuracy: 96.30%
- Val Acc: 96.67% (generalize tốt)
- Learning curve ổn định
- Deployment safe
- Dự báo chính xác trên 6 lớp lỗi

⚠️ **Điểm cần cải thiện:**
- Inclusion class yếu (90.48% F1)
- Dễ nhầm Inclusion ↔ Pitted_Surface
- **Khuyến nghị**: Thêm data augmentation khác nhau cho Inclusion hoặc thu thập thêm dữ liệu

---

## 📊 Output Directory
- **CNN Scratch Results**: `outputs/cnn_small_baseline/`
  - `best_model.pt` (32.6KB)
  - `history.csv` (20 epochs)
  - `metrics.json` (94.81% test acc)
  
- **ResNet18 Transfer Results**: `outputs/resnet18_transfer/`
  - `best_model.pt` (46.5MB)
  - `history.csv` (10 epochs)
  - `metrics.json` (96.30% test acc)
  - `confusion_matrix.png`
  - `curves.png`

## 📝 Ghi Chú
- Tất cả models được train với W&B logging
- Early stopping patience áp dụng cho cả 2 models
- Data augmentation: RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
- Test set: 270 samples (45 samples × 6 classes)
