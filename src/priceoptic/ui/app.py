"""Streamlit demo: live elasticity table, price optimizer, A/B simulator, sales feed."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("PRICEOPTIC_API_URL", "http://localhost:8120")

st.set_page_config(page_title="priceoptic", page_icon="💸", layout="wide")
st.title("💸 priceoptic")
st.caption(
    "Elasticities that update as sales arrive, constrained price optimization, A/B simulation"
)


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

tab_table, tab_opt, tab_feed = st.tabs(["Elasticities", "Optimize & simulate", "Sales feed"])

with tab_table:
    st.markdown(
        "Running estimate vs **true** elasticity (synthetic ground truth) with 95% CIs. "
        "`n_obs` counts the weeks folded so far; `priceable` is the se ≤ 0.5 gate."
    )
    st.dataframe(products, use_container_width=True, hide_index=True)
    st.caption(
        f"{int(products['priceable'].sum())} of {len(products)} products currently priceable"
    )

with tab_opt:
    pid = st.selectbox("Product", products["product_id"].tolist())
    row = products[products["product_id"] == pid].iloc[0]
    st.caption(
        f"{row['category']} — price ${row['base_price']:.2f}, cost ${row['cost']:.2f}, "
        f"elasticity {row['elasticity']} [{row['ci_lo']}, {row['ci_hi']}] from {row['n_obs']} weeks"
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

with tab_feed:
    st.markdown(
        "Post a week of sales for a product. The running fit absorbs it and the "
        "response says whether the recommendation moved, flipped direction, or the "
        "product crossed the priceable gate. Re-posting the same week is a no-op."
    )
    fpid = st.selectbox("Product", products["product_id"].tolist(), key="feed_pid")
    frow = products[products["product_id"] == fpid].iloc[0]
    c1, c2, c3 = st.columns(3)
    week = c1.number_input("Week", min_value=0, value=int(frow["n_obs"]), step=1)
    price = c2.number_input("Price", min_value=0.01, value=float(frow["base_price"]), step=0.5)
    units = c3.number_input("Units sold", min_value=0, value=100, step=10)
    if st.button("Ingest week", type="primary"):
        ri = httpx.post(
            f"{API_URL}/ingest",
            json={
                "observations": [
                    {
                        "product_id": int(fpid),
                        "week": int(week),
                        "price": price,
                        "units": int(units),
                    }
                ]
            },
            timeout=60,
        )
        if ri.status_code != 200:
            st.error(ri.json().get("detail", ri.text))
        else:
            body = ri.json()
            if body["duplicates"]:
                st.info(
                    f"Week {int(week)} was already on the ledger for product {fpid}; nothing changed."
                )
            for delta in body["deltas"]:
                b, a = delta["before"], delta["after"]
                st.write(
                    f"elasticity {b['estimate']['elasticity']} → **{a['estimate']['elasticity']}** "
                    f"(se {b['estimate']['se']} → {a['estimate']['se']}, n={a['estimate']['n_obs']})"
                )
                if a["recommendation"] is None:
                    why = a["estimate"]["note"] or "CI too wide to price on"
                    st.warning(f"Not priceable after this week: {why}")
                else:
                    br = b["recommendation"]
                    was = f"${br['recommended_price']:.2f}" if br else "not priceable"
                    st.metric(
                        "Recommended price",
                        f"${a['recommendation']['recommended_price']:.2f}",
                        delta=f"was {was}",
                    )
                flags = [
                    k
                    for k in (
                        "became_priceable",
                        "lost_priceability",
                        "direction_flipped",
                        "binding_changed",
                    )
                    if delta[k]
                ]
                if flags:
                    st.warning("Changed: " + ", ".join(flags))
