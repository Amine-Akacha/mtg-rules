"""
Builds the system prompt and user message from retrieved context chunks.

Rules chunks and card chunks are formatted into distinct sections so the
LLM can easily distinguish official rules text from card oracle text.
"""

SYSTEM_PROMPT = (
    "You are an expert Magic: The Gathering judge and rules authority. "
    "Your role is to answer rules questions and explain card interactions "
    "with precision and absolute fidelity to the official Comprehensive Rules.\n\n"
    "Guidelines:\n"
    "- Reason step by step through card interactions before reaching a conclusion.\n"
    "- Cite specific rule numbers (e.g. 'Rule 116.1a') whenever the retrieved "
    "context includes them; never invent rule numbers.\n"
    "- Clearly distinguish between what the rules say definitively and what "
    "depends on context or is genuinely ambiguous.\n"
    "- State explicitly when you lack sufficient information to give a confident "
    "ruling. An honest 'I'm not certain' is always better than a wrong answer.\n"
    "- Never fabricate card text, rule text, or rulings not present in the "
    "provided context or your verified knowledge.\n"
    "- Format complex interactions as a numbered step-by-step breakdown.\n"
    "- When a ruling could vary (e.g. multiplayer vs. two-player, different "
    "formats), note the distinction."
)


def build_prompt(question: str, context_chunks: list[dict]) -> tuple[str, str]:
    """
    Assemble the system prompt and user message from retrieved context.

    Rules chunks are shown first under a '## Relevant Rules' heading; card
    chunks follow under '## Relevant Cards'. Each rules chunk is prefixed
    with its section number when available.

    Args:
        question: The user's original question.
        context_chunks: Combined list returned by retrieval.retriever.retrieve().

    Returns:
        Tuple of (system_prompt, user_message) ready to pass to the LLM.
    """
    rules_chunks = [
        c for c in context_chunks if c.get("metadata", {}).get("source") == "rules"
    ]
    card_chunks = [
        c for c in context_chunks if c.get("metadata", {}).get("source") == "cards"
    ]

    context_parts: list[str] = []

    if rules_chunks:
        context_parts.append("## Relevant Rules\n")
        for chunk in rules_chunks:
            section = chunk.get("metadata", {}).get("section", "")
            prefix = f"[Rule {section}] " if section else ""
            context_parts.append(f"{prefix}{chunk['text'].strip()}\n")

    if card_chunks:
        context_parts.append("\n## Relevant Cards\n")
        for chunk in card_chunks:
            context_parts.append(f"{chunk['text'].strip()}\n")

    context_block = (
        "\n".join(context_parts).strip()
        if context_parts
        else "No relevant context was found in the rules or card database."
    )

    user_message = (
        f"The following context was retrieved from the official MTG Comprehensive "
        f"Rules and card database:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Please answer using the context above as your primary reference. "
        f"If the context is insufficient to answer confidently, say so clearly "
        f"and explain what additional information would be needed."
    )

    return SYSTEM_PROMPT, user_message
