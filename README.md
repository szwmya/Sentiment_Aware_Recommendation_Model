
# 📚 Deep Sentiment-Aware Recommendation System

This project implements a **Hybrid Recommendation System** that combines deep sentiment analysis with collaborative filtering. It integrates RoBERTa for semantic understanding, BiGRU for sequential text processing, Multi-Head Self-Attention for contextual importance, and DeepFM for modeling high-order feature interactions.

---

## 🚀 Model Architecture

```
Input Text (Reviews)
        │
     Tokenization
        ↓
  ┌────────────┐
  │ RoBERTa    │  ← (Contextual Embeddings)
  └────────────┘
        ↓
    BiGRU Layer


        ↓
Custom Multi-Head Self-Attention
        ↓
  Attention Pooling (Softmax weighted sum)
        ↓
     DeepFM Layers
        ↓
   Final Classifier (Logits → Predictions)
```

---

## 🧠 Key Components

- **RoBERTa**: Contextual embedding of reviews.
- **BiGRU**: Captures sequential dependencies in review text.
- **Multi-Head Self-Attention**: Learns importance of tokens dynamically.
- **DeepFM**: Captures both low- and high-order feature interactions.
- **Attention Pooling**: Condenses token-level outputs into a fixed-size vector.
- **Classifier**: Predicts review sentiment or rating class.


## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Key Libraries:**

- `transformers`
- `torch`
- `sklearn`
- `pandas`
- `numpy`
- `tqdm`

---

## 📈 Results

| Metric            | Score     |
|------------------|-----------|
| Train Accuracy    | 89.83%    |
| Validation Acc    | 87.45%    |
| Test Accuracy     | **87.10%** |
| Test Loss         | 0.3926    |

---

## 🔍 Evaluation Metrics

- Accuracy
- Cross-Entropy Loss
- Precision / Recall / F1 (Optional)
- Early Stopping with Patience = 4

---

## 🧪 How to Train

```python
from model import FullRecommendationModel

model = FullRecommendationModel()
train_model(model, train_loader, val_loader)
evaluate_model(model, test_loader)
```

---

## 📝 Notes

- Freezes most of RoBERTa, fine-tunes top 4 layers.
- Optional positional encoding can be added for MHSA.
- Supports multi-class classification (3 sentiment classes).

---

## 📌 Future Work

- Integrate user/item embeddings for personalization.
- Add contrastive learning for better representation.
- Explore larger transformer backbones (e.g., DeBERTa).
- Support for multi-modal features (e.g., images).

---

## 🧑‍💻 Author
 **Sowmiya E**
  Contact: sowmyaezhumalai14@gmail.com
 **Durwin A S**
  Contact : durwinas114@gmail.com

