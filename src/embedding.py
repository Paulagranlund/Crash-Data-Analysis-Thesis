from typing import Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_sentence_embedding_model(
    model_name: str = "KennethTM/MiniLM-L6-danish-encoder",
    verbose: bool = True,
) -> SentenceTransformer:
    """
    Load a SentenceTransformer model for BERTopic or other embedding-based tasks.

    Parameters:
    model_name : str
        Name or path of the sentence embedding model.
    verbose : bool
        Whether to print progress information.

    Returns:
    SentenceTransformer
        Loaded sentence embedding model.
    """
    if verbose:
        print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(model_name)

    if verbose:
        print(f"Loaded embedding model: {model_name}")

    return embedding_model


def generate_embeddings(
    docs: list,
    embedding_model: SentenceTransformer,
    batch_size: int = 64,
    convert_to_numpy: bool = True,
    show_progress_bar: bool = True,
    verbose: bool = True,
):
    """
    Generate document embeddings from a list of texts.

    Parameters:
    docs : list
        List of input texts.
    embedding_model : SentenceTransformer
        Loaded sentence embedding model.
    batch_size : int
        Number of documents processed per batch.
    convert_to_numpy : bool
        Whether to return numpy arrays.
    show_progress_bar : bool
        Whether to show tqdm progress bar during encoding.
    verbose : bool
        Whether to print progress information.

    Returns:
    embeddings
        Array of document embeddings.
    """
    if verbose:
        print("Generating document embeddings...")
        print("(This may take a few minutes depending on dataset size)")

    embeddings = embedding_model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=convert_to_numpy,
    )

    if verbose:
        print(f"Embedding shape: {embeddings.shape}")

    return embeddings


def load_classifier_tokenizer_and_model(
    model_name: str = "Maltehb/danish-bert-botxo",
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, int]] = None,
    label2id: Optional[Dict[int, int]] = None,
    use_fast_tokenizer: bool = True,
    verbose: bool = True,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load tokenizer and classification model for supervised BERT classification.

    Parameters:
    model_name : str
        Name or path of the pre-trained classifier backbone.
    num_labels : int, optional
        Number of output classes.
    id2label : dict, optional
        Mapping from class index to original label.
    label2id : dict, optional
        Mapping from original label to class index.
    use_fast_tokenizer : bool
        Whether to use the fast tokenizer.
    verbose : bool
        Whether to print progress information.

    Returns:
    tokenizer : AutoTokenizer
        Loaded tokenizer.
    model : AutoModelForSequenceClassification
        Loaded classification model.
    """
    if verbose:
        print("\nLoading classifier tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)

    if verbose:
        print("Loading classifier model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if verbose:
        print(f"Loaded classifier model: {model_name}")

    return tokenizer, model

"""
How to use for BERTopic:

from embedding import load_sentence_embedding_model, generate_embeddings

MODEL_NAME = "KennethTM/MiniLM-L6-danish-encoder"

embedding_model = load_sentence_embedding_model(MODEL_NAME)
embeddings = generate_embeddings(
    docs=docs,
    embedding_model=embedding_model,
    batch_size=64,
    verbose=True,
)

How to use for BERT classification:

from embedding_load import load_classifier_tokenizer_and_model

MODEL_NAME = "Maltehb/danish-bert-botxo"

tokenizer, model = load_classifier_tokenizer_and_model(
    model_name=MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
"""