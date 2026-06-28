"""Integrated Gradients token attribution for the fine-tuned classifier.

Plain functions wrap Captum's LayerIntegratedGradients to produce per-token
attributions, merge WordPiece subwords back into whole words, and aggregate the
word-level attributions across a subset of narratives into class-level token
summaries. Attributions are L2-normalised per narrative for readability.

Build the LayerIntegratedGradients object once with ``build_lig(model)`` and
pass it, together with the model, tokenizer and id2label mapping, into the
attribution functions.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients


def build_lig(model):
    """Build a LayerIntegratedGradients object over the model's embedding layer.

    The forward function uses input_ids so the embedding layer is actually
    exercised by Integrated Gradients.
    """
    model.eval()

    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    embedding_layer = model.get_input_embeddings()
    return LayerIntegratedGradients(forward_func, embedding_layer)


def attribute_text(text, target_class, model, tokenizer, lig, id2label, max_length,
                   device=None, pred_class_id=None, pred_conf=None, n_steps=20,
                   internal_batch_size=2, method="gausslegendre"):
    """Per-token IG attribution of one narrative toward target_class.

    A PAD-token baseline is used, the embedding dimension is collapsed to one
    score per token, and the scores are L2-normalised. Returns a dict with
    tokens, scores, and prediction/target metadata.
    """
    device = device or model.device
    model.eval()
    model.zero_grad()

    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id, so a PAD-token baseline cannot be "
            "created."
        )
    baseline_ids = torch.full_like(input_ids, pad_token_id).to(device)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=int(target_class),
        n_steps=n_steps,
        method=method,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    # Collapse embedding dimension to one score per token.
    token_attributions = attributions.sum(dim=-1).squeeze(0)

    # L2-normalise for readability only.
    norm = torch.norm(token_attributions)
    if norm > 0:
        token_attributions = token_attributions / norm

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    valid_len = int(attention_mask.sum().item())
    tokens = tokens[:valid_len]
    scores = token_attributions[:valid_len].detach().cpu().numpy()

    return {
        "text": text,
        "tokens": tokens,
        "scores": scores,
        "pred_class_id": int(pred_class_id) if pred_class_id is not None else None,
        "pred_class_label": id2label[int(pred_class_id)] if pred_class_id is not None else None,
        "target_class_id": int(target_class),
        "target_class_label": id2label[int(target_class)],
        "pred_conf": float(pred_conf) if pred_conf is not None else None,
        "delta": float(delta.detach().cpu().item()) if hasattr(delta, "detach") else float(delta),
    }


def merge_wordpiece_tokens(tokens, scores):
    """Merge BERT WordPiece pieces into whole words, summing their scores.

    For example ["vige", "##pligt"] becomes ["vigepligt"]. Special tokens
    ([CLS], [SEP], [PAD]) are dropped.
    """
    merged_tokens, merged_scores = [], []
    current_token, current_score = None, 0.0

    for tok, score in zip(tokens, scores):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if tok.startswith("##"):
            if current_token is not None:
                current_token += tok[2:]
                current_score += score
            else:
                # Edge case: a subword appears first.
                current_token = tok[2:]
                current_score = score
        else:
            if current_token is not None:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
            current_token = tok
            current_score = score

    if current_token is not None:
        merged_tokens.append(current_token)
        merged_scores.append(current_score)

    return merged_tokens, merged_scores


def get_correct_examples_for_class(df, class_id, n=10, sort_by_conf=True):
    """Correctly classified test rows for a class, optionally by confidence."""
    subset = df[
        (df["true_label_id"] == class_id) & (df["pred_label_id"] == class_id)
    ].copy()
    if sort_by_conf:
        subset = subset.sort_values("pred_conf", ascending=False)
    return subset.head(n)


def get_misclassified_examples_for_class(df, class_id, n=10, mode="true",
                                         sort_by_conf=True):
    """Misclassified rows for a class.

    mode='true' selects rows whose true class is class_id but whose prediction
    differs; mode='pred' selects rows predicted as class_id whose true class
    differs.
    """
    if mode == "true":
        subset = df[
            (df["true_label_id"] == class_id) & (df["pred_label_id"] != class_id)
        ].copy()
    elif mode == "pred":
        subset = df[
            (df["pred_label_id"] == class_id) & (df["true_label_id"] != class_id)
        ].copy()
    else:
        raise ValueError("mode must be 'true' or 'pred'")

    if sort_by_conf:
        subset = subset.sort_values("pred_conf", ascending=False)
    return subset.head(n)


def aggregate_attributions_for_subset(df_subset, model, tokenizer, lig, id2label,
                                      max_length, explain_target="pred",
                                      n_steps=250, internal_batch_size=2,
                                      method="gausslegendre"):
    """Average word-level IG attributions across a subset of narratives.

    explain_target='pred' explains each row's predicted class, 'true' the true
    class. Returns a dataframe with token, sum_score, count and mean_score,
    sorted by descending mean_score.
    """
    token_sum = defaultdict(float)
    token_count = defaultdict(int)

    for _, row in df_subset.iterrows():
        text = row["police_narrative"]

        if explain_target == "pred":
            target_class = int(row["pred_label_id"])
        elif explain_target == "true":
            target_class = int(row["true_label_id"])
        else:
            raise ValueError("explain_target must be 'pred' or 'true'")

        result = attribute_text(
            text=text,
            target_class=target_class,
            model=model,
            tokenizer=tokenizer,
            lig=lig,
            id2label=id2label,
            max_length=max_length,
            pred_class_id=row["pred_label_id"],
            pred_conf=row["pred_conf"],
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            method=method,
        )

        merged_tokens, merged_scores = merge_wordpiece_tokens(
            result["tokens"], result["scores"]
        )

        for tok, score in zip(merged_tokens, merged_scores):
            tok_clean = tok.strip()
            if tok_clean == "":
                continue
            token_sum[tok_clean] += float(score)
            token_count[tok_clean] += 1

    rows = [
        {
            "token": tok,
            "sum_score": token_sum[tok],
            "count": token_count[tok],
            "mean_score": token_sum[tok] / token_count[tok],
        }
        for tok in token_sum
    ]
    token_summary_df = pd.DataFrame(rows)
    if len(token_summary_df) == 0:
        return token_summary_df

    return token_summary_df.sort_values(
        "mean_score", ascending=False
    ).reset_index(drop=True)


def print_top_tokens(token_summary_df, top_k=15):
    """Print the most positive and most negative mean-attribution tokens."""
    if len(token_summary_df) == 0:
        print("No tokens found.")
        return

    print("\nTop positive tokens:")
    for _, row in token_summary_df.head(top_k).iterrows():
        print(
            f"{row['token']:25s} mean={row['mean_score']:+.4f}  "
            f"sum={row['sum_score']:+.4f}  count={int(row['count'])}"
        )

    print("\nTop negative tokens:")
    for _, row in token_summary_df.sort_values("mean_score").head(top_k).iterrows():
        print(
            f"{row['token']:25s} mean={row['mean_score']:+.4f}  "
            f"sum={row['sum_score']:+.4f}  count={int(row['count'])}"
        )


def save_token_summary(token_summary_df, class_id, output_dir, suffix):
    """Write a class-level token summary to CSV."""
    path = os.path.join(output_dir, f"class_{class_id}_{suffix}_token_summary.csv")
    token_summary_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def run_class_token_analysis(df, class_id, model, tokenizer, lig, id2label,
                             max_length, n_examples=10, mode="correct",
                             explain_target="pred", n_steps=20,
                             internal_batch_size=2, method="gausslegendre",
                             output_dir=None, top_k=15):
    """End-to-end class-level IG analysis for one class.

    mode selects which rows to explain: 'correct' for correctly classified
    examples, 'mis_true' for misclassified rows whose true class is class_id,
    and 'mis_pred' for rows predicted as class_id whose true class differs.
    explain_target chooses whether the predicted or true class is explained.
    Returns the example subset and the aggregated token summary.
    """
    if mode == "correct":
        subset = get_correct_examples_for_class(df, class_id, n=n_examples)
        suffix = f"correct_explain_{explain_target}"
    elif mode == "mis_true":
        subset = get_misclassified_examples_for_class(
            df, class_id, n=n_examples, mode="true"
        )
        suffix = f"mis_true_explain_{explain_target}"
    elif mode == "mis_pred":
        subset = get_misclassified_examples_for_class(
            df, class_id, n=n_examples, mode="pred"
        )
        suffix = f"mis_pred_explain_{explain_target}"
    else:
        raise ValueError("mode must be 'correct', 'mis_true', or 'mis_pred'")

    print("=" * 100)
    print(
        f"CLASS {class_id} | mode: {mode} | explain target: {explain_target} "
        f"| examples: {len(subset)}"
    )
    print("=" * 100)

    token_summary_df = aggregate_attributions_for_subset(
        subset,
        model=model,
        tokenizer=tokenizer,
        lig=lig,
        id2label=id2label,
        max_length=max_length,
        explain_target=explain_target,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        method=method,
    )
    print_top_tokens(token_summary_df, top_k=top_k)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_token_summary(token_summary_df, class_id, output_dir, suffix)
        subset.to_csv(
            os.path.join(output_dir, f"class_{class_id}_{suffix}_examples.csv"),
            index=False,
        )

    return subset, token_summary_df
