"""Page 5: Source Conflict Explorer displaying agreement/contradiction matrices and source-wise evidence."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore


def render_conflict_explorer_page(state: Dict[str, Any]) -> None:
    """Render Page 5: Source Conflict Explorer in Streamlit."""
    if st is None:
        return
    st.title("Cross-Source Conflict Explorer")
    last_res = state.get("last_query_result")
    if not last_res or not last_res["verification"].get("claims"):
        st.info("Run a query on Page 3 to explore cross-source conflict matrices.")
        return

    claims = last_res["verification"]["claims"]
    claim_options = {c["claim_id"]: f"{c['claim_id']}: {c['claim_text']}" for c in claims}
    selected_id = st.selectbox("Select Atomic Claim", options=list(claim_options.keys()), format_func=lambda k: claim_options[k])

    selected_claim = next(c for c in claims if c["claim_id"] == selected_id)
    conflict = selected_claim["conflict_analysis"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conflict Detected", "YES" if conflict["has_conflict"] else "NO")
    c2.metric("Conflict Type", conflict["conflict_type"])
    c3.metric("Conflict Severity (CSI)", f"{conflict['conflict_severity']:.3f}")
    c4.metric("Cross-Source Agreement", f"{conflict['cross_source_agreement']:.3f}")

    st.subheader("Source-Wise Evidence & NLI Verification Matrix")
    st.dataframe(pd.DataFrame(conflict["source_verification_table"]), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Pairwise Agreement Matrix ($1 - \text{JSD}$)")
        st.dataframe(pd.DataFrame(conflict["agreement_matrix"]), use_container_width=True)
    with col_b:
        st.markdown("#### Pairwise Polarity Contradiction Matrix ($\Gamma_{a,b}$)")
        st.dataframe(pd.DataFrame(conflict["contradiction_matrix"]), use_container_width=True)
