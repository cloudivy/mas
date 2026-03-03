import json
import openai
from core.embeddings import get_embedding, cosine_similarity, jaccard_terms
from core.agents import NODES, TASK_SPEC

ASI_DIMS = [
    {
        "key": "response_consistency",
        "label": "Response Consistency",
        "weight": 0.25,
        "desc": "Cosine similarity of current output to baseline embedding",
        "color": "#4FC3F7",
    },
    {
        "key": "reasoning_stability",
        "label": "Reasoning Stability",
        "weight": 0.25,
        "desc": "Cosine similarity vs previous step output",
        "color": "#81C784",
    },
    {
        "key": "inter_agent_agreement",
        "label": "Inter-Agent Agreement",
        "weight": 0.25,
        "desc": "Jaccard term overlap across all active node outputs",
        "color": "#FFB74D",
    },
    {
        "key": "task_adherence",
        "label": "Task Adherence",
        "weight": 0.25,
        "desc": "LLM-judge score vs original task specification",
        "color": "#CE93D8",
    },
]


def compute_asi(
    api_key: str,
    step_idx: int,
    curr_outputs: dict,
    baseline_embeddings: dict,
    prev_outputs: dict,
) -> dict:
    """
    Compute the Agent Stability Index and its four sub-dimensions.

    Returns:
        dict with keys:
            dims  (dict)  — per-dimension scores
            asi   (float) — composite weighted score
    """
    dims = {}

    # ── 1. Response Consistency ───────────────────────────────────────────────
    # Cosine similarity between current output embedding and the stored baseline
    rcs = []
    for node in NODES:
        nid = node["id"]
        if curr_outputs.get(nid) and baseline_embeddings.get(nid):
            emb = get_embedding(api_key, curr_outputs[nid])
            rcs.append(cosine_similarity(emb, baseline_embeddings[nid]))
    dims["response_consistency"] = round(sum(rcs) / len(rcs), 4) if rcs else 1.0

    # ── 2. Reasoning Stability ────────────────────────────────────────────────
    # Cosine similarity between current and previous step outputs per node
    if prev_outputs:
        rs = []
        for node in NODES:
            nid = node["id"]
            if curr_outputs.get(nid) and prev_outputs.get(nid):
                e1 = get_embedding(api_key, curr_outputs[nid])
                e2 = get_embedding(api_key, prev_outputs[nid])
                rs.append(cosine_similarity(e1, e2))
        dims["reasoning_stability"] = round(sum(rs) / len(rs), 4) if rs else 1.0
    else:
        dims["reasoning_stability"] = 1.0

    # ── 3. Inter-Agent Agreement ──────────────────────────────────────────────
    # Pairwise Jaccard term overlap across all active node outputs
    outputs = [curr_outputs[n["id"]] for n in NODES if curr_outputs.get(n["id"])]
    if len(outputs) >= 2:
        total, count = 0.0, 0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                total += jaccard_terms(outputs[i], outputs[j])
                count += 1
        dims["inter_agent_agreement"] = round(total / count, 4) if count else 1.0
    else:
        dims["inter_agent_agreement"] = 1.0

    # ── 4. Task Adherence ─────────────────────────────────────────────────────
    # LLM-as-judge scoring; called every 2 steps to reduce API cost
    if step_idx % 2 == 0:
        client = openai.OpenAI(api_key=api_key)
        combined = "\n\n".join(outputs)[:800]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator. "
                            "Return ONLY valid JSON: {\"score\": <float 0.0-1.0>} "
                            "where 1.0 = perfectly on-task. No other text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task specification:\n{TASK_SPEC}\n\n"
                            f"Agent outputs at step {step_idx + 1}:\n{combined}\n\n"
                            "Score task adherence:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=30,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            dims["task_adherence"] = round(max(0.0, min(1.0, float(parsed["score"]))), 4)
        except Exception:
            dims["task_adherence"] = 0.75
    else:
        dims["task_adherence"] = None  # interpolated in UI

    # ── Composite ASI ─────────────────────────────────────────────────────────
    ta = dims["task_adherence"] if dims["task_adherence"] is not None else 0.75
    asi = round(
        dims["response_consistency"]  * 0.25
        + dims["reasoning_stability"] * 0.25
        + dims["inter_agent_agreement"] * 0.25
        + ta * 0.25,
        4,
    )

    return {"dims": dims, "asi": asi}
