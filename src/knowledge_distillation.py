"""
knowledge_distillation.py — NLP Knowledge Distillation Pipeline
================================================================
Step 1: Use Claude (teacher) to generate a gold-standard labelled
        hotel review dataset of N samples with rich annotations.
Step 2: Fine-tune a smaller HuggingFace model (student) on that dataset.
        Recommended: distilbert-base-uncased (66M params, 97% of BERT performance)

Why this matters:
  • Claude API: 200ms latency, $0.003/1k tokens, data leaves your infra
  • Fine-tuned DistilBERT: 8ms latency, zero API cost, runs on your hardware
  • Accuracy gap after distillation: typically < 2% on domain-specific tasks

Usage
-----
  # Generate dataset (requires ANTHROPIC_API_KEY):
  python src/knowledge_distillation.py --generate --n 500

  # Fine-tune student model (requires GPU recommended, CPU ok for small n):
  python src/knowledge_distillation.py --finetune

  # Full pipeline:
  python src/knowledge_distillation.py --generate --n 500 --finetune
"""

import os, json, time, argparse, random
from pathlib import Path
import pandas as pd

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

DATASET_PATH = Path("data/distillation_dataset.csv")
MODEL_DIR    = Path("models/distilbert_hotel_sentiment")

# ─────────────────────────────────────────────────────────────────────────────
# REVIEW TEMPLATES FOR DIVERSE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
GENERATION_PROMPTS = [
    # Positive
    "a glowing review mentioning exceptional spa service and stunning views",
    "a 5-star review about impeccable concierge service for a business trip",
    "a positive review highlighting excellent value and comfortable rooms",
    "an enthusiastic review about a perfect honeymoon experience",
    "a warm review praising the attentive staff and delicious breakfast",
    # Negative
    "a frustrated review about a broken air conditioning unit and unresponsive staff",
    "an angry review about being overcharged and poor room cleanliness",
    "a disappointed review about a cancelled reservation with no notification",
    "a scathing review with sarcasm about 'wonderful' service that was actually terrible",
    "a negative review about noisy rooms and slow room service",
    # Neutral / Mixed
    "a mixed review: great location but average food and slow wifi",
    "a balanced review praising the pool but criticising the parking",
    "a neutral review giving average scores across all categories",
    "a nuanced review: some great things but too expensive for what it offers",
]

GENERATION_SYSTEM = """You are generating training data for a hotel sentiment classifier.
Generate a realistic hotel guest review matching the description.
Return ONLY a valid JSON object with no preamble or markdown fences:
{
  "text": "The review text (40-120 words, realistic guest voice)",
  "label": "Positive" | "Neutral" | "Negative",
  "polarity": float from -1.0 to 1.0,
  "confidence": float 0.0-1.0,
  "aspects": {
    "room": float or null,
    "service": float or null,
    "food": float or null,
    "value": float or null,
    "location": float or null
  },
  "sarcasm_flag": boolean,
  "themes": ["theme1", "theme2"]
}
Make the text sound like a real hotel review — include specific details, vary sentence structure."""


def generate_dataset(n: int = 500) -> pd.DataFrame:
    """
    Generate n labelled hotel reviews using Claude API.
    """
    if not HAS_ANTHROPIC:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable.")

    client  = Anthropic(api_key=api_key)
    records = []
    prompts = GENERATION_PROMPTS * (n // len(GENERATION_PROMPTS) + 1)
    random.shuffle(prompts)

    print(f"Generating {n} labelled reviews via Claude API…")
    for i, prompt in enumerate(prompts[:n]):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                system=GENERATION_SYSTEM,
                messages=[{"role":"user","content": f"Description: {prompt}"}],
            )
            raw = resp.content[0].text.strip()
            # Strip any accidental markdown
            import re
            raw = re.sub(r"```json|```","",raw).strip()
            record = json.loads(raw)
            record["source"] = "claude_generated"
            record["prompt_template"] = prompt
            records.append(record)

            if (i+1) % 50 == 0:
                print(f"  {i+1}/{n} generated…")
            time.sleep(0.3)   # Rate-limit courtesy

        except Exception as e:
            print(f"  Warning — skipping record {i}: {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(DATASET_PATH, index=False)
    print(f"\n✓ Dataset saved: {DATASET_PATH}  ({len(df)} rows)")
    print(f"  Label distribution:\n{df['label'].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def finetune_student(
    dataset_path: str  = str(DATASET_PATH),
    output_dir:   str  = str(MODEL_DIR),
    model_name:   str  = "distilbert-base-uncased",
    epochs:       int  = 3,
    batch_size:   int  = 16,
    lr:           float= 2e-5,
):
    """
    Fine-tune DistilBERT on the gold-standard dataset.
    Requires: transformers, torch, datasets
    """
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
        )
        from datasets import Dataset
        import torch
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as e:
        print(f"Missing: {e}")
        print("Install: pip install transformers torch datasets")
        return

    print(f"\nFine-tuning {model_name} on hotel sentiment dataset…")
    df = pd.read_csv(dataset_path)
    df = df[["text","label"]].dropna()
    df["label_id"] = df["label"].map(LABEL_MAP)

    # Train/validation split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42,
                                         stratify=df["label_id"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenise(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text","label_id"]].rename(
        columns={"label_id":"labels"})).map(tokenise, batched=True)
    val_ds   = Dataset.from_pandas(val_df[["text","label_id"]].rename(
        columns={"label_id":"labels"})).map(tokenise, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0:"Negative",1:"Neutral",2:"Positive"},
        label2id=LABEL_MAP,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir          = output_dir,
        num_train_epochs    = epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate       = lr,
        weight_decay        = 0.01,
        evaluation_strategy="epoch",
        save_strategy       ="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps       = 10,
        report_to           = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        data_collator   = DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    print(f"\n✓ Fine-tuned model saved → {output_dir}")
    print(f"  Validation accuracy: {metrics.get('eval_accuracy',0):.3f}")
    print(f"  Validation F1:       {metrics.get('eval_f1_macro',0):.3f}")

    # Save metadata
    meta = {
        "model_name":    model_name,
        "output_dir":    output_dir,
        "training_rows": len(train_df),
        "val_rows":      len(val_df),
        "epochs":        epochs,
        **metrics,
    }
    with open(os.path.join(output_dir,"metadata.json"),"w") as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate",  action="store_true", help="Generate dataset via Claude")
    parser.add_argument("--finetune",  action="store_true", help="Fine-tune DistilBERT")
    parser.add_argument("--n",         type=int, default=500, help="Number of reviews to generate")
    parser.add_argument("--model",     type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs",    type=int, default=3)
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    if args.generate:
        generate_dataset(n=args.n)
    if args.finetune:
        finetune_student(model_name=args.model, epochs=args.epochs)
    if not args.generate and not args.finetune:
        print("Usage: python knowledge_distillation.py --generate [--n 500] [--finetune]")
