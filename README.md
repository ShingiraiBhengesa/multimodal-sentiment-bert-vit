# Multimodal Sentiment Analysis (BERT + ViT)

Predict sentiment (positive/negative) from review text + product image.
- Text encoder: `bert-base-uncased`
- Image encoder: `google/vit-base-patch16-224`
- Fusion: concat([CLS_text, CLS_image]) → MLP

## Run (Colab)
1) Install deps: `pip install -r requirements.txt`
2) Prepare `data/train.csv`, `val.csv`, `test.csv` with columns: `text,image_path,label`
3) Train: `python -m src.train_text`, `python -m src.train_image`, `python -m src.train_fusion`
4) Evaluate: `python -m src.eval`
