"""Streamlit demo: elasticity table, price optimizer, A/B simulator."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("PRICEOPTIC_API_URL", "http://localhost:8120")

st.set_page_config(page_title="priceoptic", page_icon="💸", layout="wide")
st.title("💸 priceoptic")
st.caption("Elasticity estimation with CIs, constrained price optimization, A/B revenue simulation")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

r = httpx.get(f"{API_URL}/products", timeout=30)
if r.status_code != 200:
    st.warning(r.json().get("detail", r.text))
    st.stop()
products = pd.DataFrame(r.json())

tab_table, tab_opt = st.tabs(["Elasticities", "Optimize & simulate"])

with tab_table:
    st.markdown("Estimated vs **true** elasticity (synthetic ground truth) with 95% CIs")
    st.dataframe(products, use_container_width=True, hide_index=True)

with tab_opt:
    pid = st.selectbox("Product", products["product_id"].tolist())
    row = products[products["product_id"] == pid].iloc[0]
    st.caption(
        f"{row['category']} — price ${row['base_price']:.2f}, cost ${row['cost']:.2f}, "
        f"elasticity {row['elasticity']} [{row['ci_lo']}, {row['ci_hi']}]"
    )
    ro = httpx.get(f"{API_URL}/optimize/{pid}", timeout=30)
    if ro.status_code != 200:
        st.warning(ro.json().get("detail", ro.text))
    else:
        rec = ro.json()
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Recommended price",
            f"${rec['recommended_price']:.2f}",
            delta=f"{rec['change_pct']:+.1%}",
        )
        c2.metric("Expected profit lift", f"{rec['expected_profit_lift_pct']:+.1%}")
        c3.metric("Binding constraint", rec["binding_constraint"])

        candidate = st.slider(
            "Simulate a candidate price",
            float(row["base_price"]) * 0.7,
            float(row["base_price"]) * 1.3,
            float(rec["recommended_price"]),
        )
        if st.button("Run A/B simulation", type="primary"):
            rs = httpx.post(
                f"{API_URL}/simulate",
                json={"product_id": int(pid), "candidate_price": candidate},
                timeout=60,
            )
            if rs.status_code != 200:
                st.error(rs.json().get("detail", rs.text))
            else:
                sim = rs.json()
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Profit diff ({sim['weeks']}wk)", f"${sim['profit_diff_mean']:,.0f}")
                c2.metric(
                    "90% interval",
                    f"${sim['profit_diff_p5']:,.0f} … ${sim['profit_diff_p95']:,.0f}",
                )
                c3.metric("P(profit improves)", f"{sim['prob_profit_positive']:.0%}")
                if sim["prob_profit_positive"] < 0.7:
                    st.warning("Meaningful downside risk — the elasticity CI is doing its job.")
