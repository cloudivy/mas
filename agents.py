import openai

TASK_SPEC = """Synthesise the academic literature on this question:
"How does Retrieval-Augmented Generation (RAG) improve factual accuracy in LLMs compared to standard fine-tuning?"
Your synthesis must cover:
1. Mechanism of RAG vs fine-tuning
2. Key empirical findings with evidence
3. Limitations of each approach
4. A concrete practitioner recommendation."""

NODES = [
    {
        "id": "retriever",
        "label": "Retriever",
        "role": "Information Retrieval Node",
        "instruction": (
            "Surface relevant facts, papers, and evidence from prior context. "
            "Extract key claims precisely. Do not summarise — surface raw material for downstream nodes."
        ),
        "color": "#4FC3F7",
    },
    {
        "id": "summarizer",
        "label": "Summarizer",
        "role": "Synthesis Node",
        "instruction": (
            "Take retrieved material and produce a structured summary. "
            "Group related findings. Use consistent terminology. Preserve the original task framing."
        ),
        "color": "#81C784",
    },
    {
        "id": "critic",
        "label": "Critic",
        "role": "Evaluation Node",
        "instruction": (
            "Critically evaluate the summary for gaps, errors, or deviation from the original task. "
            "Flag any claims not grounded in evidence. Be rigorous and specific."
        ),
        "color": "#FFB74D",
    },
    {
        "id": "compositor",
        "label": "Compositor",
        "role": "Output Node",
        "instruction": (
            "Compose the final synthesis report using all prior node outputs. "
            "Ensure all four required sections are addressed. Stay tightly anchored to the original task."
        ),
        "color": "#CE93D8",
    },
]


def execute_node(
    api_key: str,
    node: dict,
    step_idx: int,
    history: list,
    total_steps: int,
    apply_mitigation: bool = False,
) -> str:
    """Run a single LangGraph node and return its text output."""
    client = openai.OpenAI(api_key=api_key)

    # Episodic Memory Consolidation: keep only last 4 history entries
    hist_slice = history[-4:] if apply_mitigation else history
    hist_text = (
        "\n\n---\n\n".join(
            f"[Step {h['step']} · {h['node_label']}]\n{h['output']}"
            for h in hist_slice
        )
        if hist_slice
        else "No prior state."
    )

    # Behavioral Anchoring: re-inject task spec every step
    anchor = (
        f"\n\n⚓ TASK ANCHOR — re-read every step:\n{TASK_SPEC}\n"
        if apply_mitigation
        else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are the {node['role']} in a LangGraph research synthesis pipeline.\n"
                f"{node['instruction']}{anchor}\n"
                f"Step {step_idx + 1}/{total_steps}. "
                "Be concise (max 120 words). Never lose sight of the original task."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL TASK:\n{TASK_SPEC}\n\n"
                f"GRAPH STATE HISTORY:\n{hist_text}\n\n"
                f"Now produce your node output as {node['label']}:"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=350,
    )
    return response.choices[0].message.content.strip()
