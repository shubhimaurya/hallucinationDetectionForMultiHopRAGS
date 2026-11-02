"""Page 4: Claim-Level Hallucination Analysis displaying color-coded verification cards per atomic claim."""
from __future__ import annotations

from typing import Any, Dict, List

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

CLASSIFICATION_BADGE_COLORS = {
    "SUPPORTED": ("#1b7f4c", "#e6f4ea", "GREEN — SUPPORTED"),
    "HALLUCINATED": ("#b91c1c", "#fce8e6", "RED — CONTRADICTED / HALLUCINATED"),
    "UNCERTAIN": ("#b45309", "#fef7e0", "YELLOW — UNCERTAIN / ABSTAINED"),
}


def format_claim_cards(verification_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform verified claims into structured card dictionaries with color styling."""
    cards: List[Dict[str, Any]] = []
    for clm in verification_output.get("claims", []):
        conf = clm["confidence_result"]
        conflict = clm["conflict_analysis"]
        judge = clm["judge_result"]
        status = clm["final_classification"]
        fg_color, bg_color, badge_label = CLASSIFICATION_BADGE_COLORS.get(
            status, ("#374151", "#f3f4f6", status)
        )
        cards.append(
            {
                "claim_id": clm["claim_id"],
                "claim_text": clm["claim_text"],
                "final_classification": status,
                "fine_grained_subtype": clm["fine_grained_subtype"],
                "badge_label": badge_label,
                "fg_color": fg_color,
                "bg_color": bg_color,
                "nli_entailment_score": conf["E_entailment"],
                "nli_contradiction_score": conf["K_contradiction"],
                "evidence_coverage": conf["evidence_coverage"],
                "source_agreement": conf["A_agreement"],
                "conflict_severity": conflict["conflict_severity"],
                "llm_judge_invoked": judge["invoked"],
                "llm_judge_result": judge["verification_decision"],
                "confidence_score": clm["calibrated_confidence"],
                "explanation": clm["explanation_text"],
            }
        )
    return cards


def render_claim_analysis_page(state: Dict[str, Any]) -> None:
    """Render Page 4: Claim-Level Hallucination Analysis in Streamlit."""
    if st is None:
        return
    st.title("Atomic Claim-Level Hallucination Analysis")
    last_res = state.get("last_query_result")
    if not last_res:
        st.info("Run a query on Page 3 first to inspect extracted atomic claims.")
        return

    ver = last_res["verification"]
    cards = format_claim_cards(ver)
    st.write(
        f"**Extracted Atomic Claims:** {len(cards)} | "
        f"**Supported:** {ver['supported_claims']} | "
        f"**Hallucinated:** {ver['hallucinated_claims']} | "
        f"**Uncertain:** {ver['uncertain_claims']}"
    )

    for card in cards:
        st.markdown(
            f"""
            <div style="border-left: 6px solid {card['fg_color']}; background-color: {card['bg_color']};
                        padding: 14px 18px; border-radius: 6px; margin-bottom: 14px;">
                <h4 style="margin:0; color:{card['fg_color']};">
                    [{card['claim_id']}] {card['badge_label']} ({card['fine_grained_subtype']})
                </h4>
                <p style="font-size:1.05rem; margin: 8px 0;"><strong>Claim:</strong> {card['claim_text']}</p>
                <p style="margin: 4px 0; font-size:0.92rem;">
                    <strong>NLI Entailment (E):</strong> {card['nli_entailment_score']:.3f} &nbsp;|&nbsp;
                    <strong>NLI Contradiction (K):</strong> {card['nli_contradiction_score']:.3f} &nbsp;|&nbsp;
                    <strong>Evidence Coverage:</strong> {card['evidence_coverage']:.2%} &nbsp;|&nbsp;
                    <strong>Source Agreement (A):</strong> {card['source_agreement']:.3f} &nbsp;|&nbsp;
                    <strong>LLM Judge:</strong> {card['llm_judge_result']} &nbsp;|&nbsp;
                    <strong>Calibrated Confidence:</strong> {card['confidence_score']:.3f}
                </p>
                <p style="margin: 4px 0; font-size:0.88rem; color:#333;">
                    <em>{card['explanation']}</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
