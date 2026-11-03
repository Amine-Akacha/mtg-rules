"""
HuggingFace dataset loading and formatting for MTG cards.

Loads the MechaCroc/magic-the-gathering dataset (train_clean split) and
converts each card record into a single readable text block suitable for
embedding and retrieval.
"""

from config import settings


def format_card(card: dict) -> str:
    """
    Serialise a card record into a human-readable text block.

    Tries common field name variants (e.g. 'mana_cost' vs 'cost') so the
    function is robust to minor schema changes in the upstream dataset.
    Power/Toughness is only included when both values are present.

    Args:
        card: A single row from the HuggingFace dataset.

    Returns:
        A pipe-delimited string describing the card.
    """

    def get(*keys: str, default: str = "") -> str:
        """Return the first non-None value found among the given keys."""
        for key in keys:
            val = card.get(key)
            if val is not None:
                return str(val).strip()
        return default

    name = get("name", default="Unknown")
    cost = get("mana_cost", "cost")
    type_line = get("type_line", "type")
    oracle_text = get("oracle_text", "text")
    power = card.get("power")
    toughness = card.get("toughness")

    parts = [
        f"Name: {name}",
        f"Cost: {cost}",
        f"Type: {type_line}",
        f"Text: {oracle_text}",
    ]

    if power is not None and toughness is not None:
        parts.append(f"Power/Toughness: {power}/{toughness}")

    return " | ".join(parts)


def load_card_chunks() -> list[dict]:
    """
    Download (or load from cache) the MTG card dataset and format every card.

    Only the split defined by settings.CARDS_SPLIT is loaded.

    Returns:
        List of chunk dicts, each with 'text' and 'metadata' keys.

    Raises:
        ValueError: When the requested split does not exist in the dataset.
    """
    from datasets import load_dataset  # heavy import; only needed at ingest time

    dataset = load_dataset(settings.CARDS_DATASET, split=settings.CARDS_SPLIT)

    chunks: list[dict] = []
    for card in dataset:
        text = format_card(card)
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "source": "cards",
                    "name": card.get("name", "Unknown"),
                },
            }
        )

    return chunks
