"""
Live Correction Engine (challenge version)
==========================================
sense → compare → decide → natural whisper

Improvements over the notebook MVP:
  1) spaCy NER for real entities (ORG, GPE, MONEY, PERCENT, CARDINAL, ...)
  2) Numeric tolerance — flag only if |user - truth| exceeds a relative/absolute gap
  3) Natural "whisper" templates instead of raw set dumps
  4) needs_interrupt aligns with advice (sim far OR real fact mismatch)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    import spacy
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "spaCy is required. Install with:\n"
        "  pip install spacy && python -m spacy download en_core_web_sm"
    ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_TOKEN = re.compile(
    r"(?P<sign>[-+])?"
    r"(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<unit>%|k|K|m|M|b|B)?"
)


def parse_number_token(token: str) -> float | None:
    """Turn '40%', '1,000', '2.5k' into a float (percent kept as 40.0, k→*1000)."""
    token = token.strip().replace(",", "")
    m = _NUM_TOKEN.fullmatch(token) or _NUM_TOKEN.search(token)
    if not m:
        return None
    value = float(m.group("value").replace(",", ""))
    if m.group("sign") == "-":
        value = -value
    unit = (m.group("unit") or "").lower()
    if unit == "%":
        return value  # compare percent points as-is
    if unit == "k":
        return value * 1_000
    if unit == "m":
        return value * 1_000_000
    if unit == "b":
        return value * 1_000_000_000
    return value


def numbers_differ(
    a: float,
    b: float,
    *,
    abs_tol: float = 0.5,
    rel_tol: float = 0.05,
) -> bool:
    """
    True if a and b are meaningfully different.
    abs_tol: ignore tiny gaps (e.g. 1000 vs 1000.2)
    rel_tol: also allow ~5% relative gap before flagging
    """
    gap = abs(a - b)
    scale = max(abs(a), abs(b), 1e-9)
    return gap > abs_tol and gap / scale > rel_tol


@dataclass
class EntityHit:
    text: str
    label: str  # spaCy label or "NUMBER"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LiveCorrectionEngine:
    """
    Product-shaped correction module.

    process(ground_truth, user_utterance) → dict with
      similarity, advice, whisper, needs_interrupt, details
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
        spacy_model: str = "en_core_web_sm",
        abs_tol: float = 0.5,
        rel_tol: float = 0.05,
    ):
        self.model = SentenceTransformer(embedding_model_name)
        self.threshold = threshold
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as e:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. Run:\n"
                f"  python -m spacy download {spacy_model}"
            ) from e

    # --- extraction ---------------------------------------------------------

    def extract_entities(self, text: str) -> list[EntityHit]:
        """spaCy NER + leftover bare numbers the NER might miss."""
        doc = self.nlp(text)
        hits: list[EntityHit] = []
        covered = set()

        useful = {
            "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
            "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
            "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
            "ORDINAL", "CARDINAL",
        }
        for ent in doc.ents:
            if ent.label_ in useful:
                hits.append(EntityHit(ent.text, ent.label_))
                covered.add(ent.text.lower())

        # Also catch plain digits / percents not tagged
        for m in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", text):
            tok = m.group(0)
            if tok.lower() not in covered:
                hits.append(EntityHit(tok, "NUMBER"))
                covered.add(tok.lower())

        return hits

    # --- decide -------------------------------------------------------------

    def _unique_numbers(self, hits: list[EntityHit]) -> list[tuple[str, float]]:
        """Parse numbers; keep one display form per value (prefer tokens with %)."""
        best: dict[float, str] = {}
        order: list[float] = []
        for h in hits:
            val = parse_number_token(h.text)
            if val is None:
                continue
            key = round(val, 6)
            if key not in best:
                order.append(key)
                best[key] = h.text
            elif "%" in h.text and "%" not in best[key]:
                best[key] = h.text
        return [(best[k], float(k)) for k in order]

    def compare_numbers(
        self, truth_hits: list[EntityHit], user_hits: list[EntityHit]
    ) -> list[dict[str, Any]]:
        """
        Pair numbers by order of appearance (toy alignment for short claims).
        Flag pairs that exceed numeric tolerance.
        """
        t_nums = self._unique_numbers(truth_hits)
        u_nums = self._unique_numbers(user_hits)

        diffs = []
        for (t_txt, t_val), (u_txt, u_val) in zip(t_nums, u_nums):
            if numbers_differ(t_val, u_val, abs_tol=self.abs_tol, rel_tol=self.rel_tol):
                diffs.append({
                    "truth": t_txt,
                    "user": u_txt,
                    "truth_value": t_val,
                    "user_value": u_val,
                })
        # Extra / missing counts
        if len(u_nums) > len(t_nums):
            for u_txt, u_val in u_nums[len(t_nums):]:
                diffs.append({"truth": None, "user": u_txt, "truth_value": None, "user_value": u_val})
        if len(t_nums) > len(u_nums):
            for t_txt, t_val in t_nums[len(u_nums):]:
                diffs.append({"truth": t_txt, "user": None, "truth_value": t_val, "user_value": None})
        return diffs

    def compare_named_entities(
        self, truth_hits: list[EntityHit], user_hits: list[EntityHit]
    ) -> tuple[set[str], set[str]]:
        """Non-numeric entity text bags (case-insensitive compare, keep display form)."""
        def bag(hits: list[EntityHit]) -> dict[str, str]:
            out = {}
            for h in hits:
                if parse_number_token(h.text) is not None:
                    continue
                if h.label in {"PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL", "NUMBER"}:
                    continue
                out[h.text.lower()] = h.text
            return out

        t, u = bag(truth_hits), bag(user_hits)
        missing = {t[k] for k in t.keys() - u.keys()}
        wrong = {u[k] for k in u.keys() - t.keys()}
        return missing, wrong

    def build_whisper(
        self,
        *,
        num_diffs: list[dict[str, Any]],
        missing: set[str],
        wrong: set[str],
        meaning_far: bool,
    ) -> str:
        """Natural short 'whisper' a live assistant could speak."""
        parts: list[str] = []

        for d in num_diffs:
            if d["truth"] is not None and d["user"] is not None:
                parts.append(
                    f"Quick check — you said {d['user']}, but the correct figure is {d['truth']}."
                )
            elif d["truth"] is not None:
                parts.append(f"Don't forget the figure {d['truth']}.")
            elif d["user"] is not None:
                parts.append(f"{d['user']} isn't in the source material.")

        if wrong and missing:
            # Best-effort pairwise hint
            w = sorted(wrong)
            m = sorted(missing)
            for wi, mi in zip(w, m):
                parts.append(f"Did you mean {mi} instead of {wi}?")
            for wi in w[len(m):]:
                parts.append(f"{wi} doesn't match the source.")
            for mi in m[len(w):]:
                parts.append(f"You may have missed {mi}.")
        else:
            for wi in sorted(wrong):
                parts.append(f"{wi} doesn't match the source.")
            for mi in sorted(missing):
                parts.append(f"You may have missed {mi}.")

        if not parts and meaning_far:
            parts.append(
                "That drifted from the source meaning — want to restate it more carefully?"
            )

        if not parts:
            return "Sounds aligned with the source."

        return " ".join(parts)

    # --- public API ---------------------------------------------------------

    def process(self, ground_truth: str, user_utterance: str) -> dict[str, Any]:
        emb_truth = self.model.encode(ground_truth)
        emb_user = self.model.encode(user_utterance)
        sim = float(cosine_similarity([emb_truth], [emb_user])[0][0])

        truth_hits = self.extract_entities(ground_truth)
        user_hits = self.extract_entities(user_utterance)

        num_diffs = self.compare_numbers(truth_hits, user_hits)
        missing, wrong = self.compare_named_entities(truth_hits, user_hits)

        meaning_far = sim < self.threshold
        fact_mismatch = bool(num_diffs or missing or wrong)
        needs_interrupt = meaning_far or fact_mismatch

        whisper = self.build_whisper(
            num_diffs=num_diffs,
            missing=missing,
            wrong=wrong,
            meaning_far=meaning_far,
        )

        if needs_interrupt:
            advice = "Correction suggested. " + whisper
        else:
            advice = "Sounds correct. " + whisper

        return {
            "similarity": sim,
            "advice": advice,
            "whisper": whisper,
            "needs_interrupt": needs_interrupt,
            "details": {
                "truth_entities": [h.__dict__ for h in truth_hits],
                "user_entities": [h.__dict__ for h in user_hits],
                "numeric_diffs": num_diffs,
                "missing_names": sorted(missing),
                "wrong_names": sorted(wrong),
                "meaning_far": meaning_far,
            },
        }


if __name__ == "__main__":
    engine = LiveCorrectionEngine()
    demo = engine.process(
        "Our AI platform reduces costs by 40%",
        "Our AI platform increases costs by 30%",
    )
    print(demo["whisper"])
    print("interrupt:", demo["needs_interrupt"], "sim:", round(demo["similarity"], 3))
