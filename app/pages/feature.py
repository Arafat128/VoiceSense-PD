"""Feature lab — score a spreadsheet row, not a microphone.

Three ways in, picked with the control at the top of this screen:

- Demo hold-out: a row the model did not train on, with the true label shown after.
- CSV upload: your own table. The column names must match the chosen dataset.
- Manual: type the top features. The rest can be filled with training averages.

Scoring is src/predict.py. Models load once through src/models/registry.py.
Training those models is training/train_tables.py.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.widgets import render_everyday, render_explanation, render_flash, set_flash
from src.batch import BatchError, read_csv_bytes, score_table, summary, validate
from src.config import DATASETS, DISCLAIMER, feature_meaning
from src.models.registry import list_available_models, load_holdout_samples, load_model
from src.predict import predict_from_features


@st.fragment
def feature_page() -> None:
    render_flash()
    st.markdown('<div class="vs-section">Feature-table screening</div>', unsafe_allow_html=True)
    models = list_available_models()
    ds_keys = sorted(
        {
            m["dataset_key"]
            for m in models
            if m["dataset_key"] in DATASETS
        }
    )
    if not ds_keys:
        st.error("No feature models found. Run `python -m training.train_tables`")
    else:
        dataset_labels = {k: DATASETS.get(k, {}).get("display_name", k) for k in ds_keys}
        dataset_key = st.selectbox(
            "Dataset model",
            options=ds_keys,
            format_func=lambda k: dataset_labels.get(k, k),
            key="feat_ds",
        )
        model_options = sorted(
            {m["model_name"] for m in models if m["dataset_key"] == dataset_key}
        )
        default_ix = model_options.index("random_forest") if "random_forest" in model_options else 0
        model_name = st.selectbox("Classifier", model_options, index=default_ix, key="feat_model")
        pack = load_model(dataset_key, model_name)
        feature_names = pack["feature_names"]
        st.caption(
            f"{pack.get('display_name')} · {len(feature_names)} features · {pack.get('language')}"
        )

        feature_mode = st.segmented_control(
            "Feature input",
            ["Demo hold-out", "CSV upload", "Manual"],
            default="Demo hold-out",
            key="feature_mode",
            required=True,
            label_visibility="collapsed",
        )
        sample_features = None
        sample_meta: dict[str, Any] = {}

        if feature_mode == "Demo hold-out":
            try:
                holdout = load_holdout_samples(dataset_key)
                options = [
                    f"{r.sample_id} | {r.subject_id} | true={'PD' if int(r.true_label)==1 else 'HC'}"
                    for r in holdout.itertuples()
                ]
                choice = st.selectbox("Sample", options, key="hold_sel")
                row = holdout.iloc[options.index(choice)]
                sample_meta = {
                    "sample_id": row["sample_id"],
                    "true_label": int(row["true_label"]),
                }
                sample_features = row[feature_names]
                st.dataframe(pd.DataFrame(sample_features).T, width="stretch", hide_index=True)
            except FileNotFoundError as e:
                st.error(str(e))

        elif feature_mode == "CSV upload":
            render_batch(dataset_key, model_name, feature_names)
            up_df = st.session_state.get("batch_df")
            if up_df is not None and st.session_state.get("batch_ok"):
                row_ix = st.number_input(
                    "Explain one row in detail (row number)", 1, len(up_df), 1, key="csv_row"
                )
                sample_features = up_df.iloc[int(row_ix) - 1][feature_names]
                sample_meta = {"source": "csv", "row": int(row_ix)}

        elif feature_mode == "Manual":
            imp = pack.get("feature_importances") or []
            top_feats = [x["feature"] for x in imp[:10]] if imp else feature_names[:10]
            medians = pack.get("train_medians") or {}
            cols = st.columns(2)
            manual_vals = {}
            for i, feat in enumerate(top_feats):
                with cols[i % 2]:
                    manual_vals[feat] = st.number_input(
                        feat,
                        value=float(medians.get(feat, 0.0)),
                        format="%.6f",
                        help=feature_meaning(feat),
                        key=f"man_{feat}",
                    )
            if st.checkbox("Fill remaining with training medians", key="man_ok"):
                full = {f: float(medians.get(f, 0.0)) for f in feature_names}
                full.update(manual_vals)
                sample_features = pd.Series(full)
                sample_meta = {"source": "manual"}

        if st.button("Run feature screening", type="primary", key="run_feat"):
            if sample_features is None:
                set_flash("error", "Feature screening failed: select or enter a sample first.")
                st.rerun(scope="fragment")
            else:
                try:
                    out = predict_from_features(dataset_key, sample_features, model_name, top_k=12)
                    st.session_state["last_feat_result"] = out
                    st.session_state["last_feat_meta"] = sample_meta
                    set_flash(
                        "success",
                        f"Feature screening succeeded: {out.get('prediction', 'done')}",
                    )
                    st.rerun(scope="fragment")
                except Exception as e:  # noqa: BLE001
                    set_flash("error", f"Feature screening failed: {e}")
                    st.rerun(scope="fragment")

        out = st.session_state.get("last_feat_result")
        if out:
            from src.plain_language import build_feature_plain

            if out["pred_label"] == 1:
                st.error(f"### {out['prediction']}")
            elif out["pred_label"] == -1:
                st.info(f"### {out['prediction']}")
            else:
                st.success(f"### {out['prediction']}")
            render_everyday(build_feature_plain(out))
            st.markdown('<div class="vs-section">Technical details</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric("P(PD)", f"{out['probability_pd']:.1%}")
            b.metric("P(Healthy)", f"{out['probability_hc']:.1%}")
            meta = st.session_state.get("last_feat_meta") or {}
            if meta.get("true_label") is not None:
                true = "PD" if meta["true_label"] == 1 else "HC"
                if out["pred_label"] == -1:
                    st.write(f"Hold-out true **{true}** · model made no call (middle band)")
                else:
                    match = meta["true_label"] == out["pred_label"]
                    st.write(f"Hold-out true **{true}** · match **{'Yes' if match else 'No'}**")
            render_explanation(out["explanation"])
            st.warning(DISCLAIMER)


def render_batch(dataset_key: str, model_name: str, feature_names: list[str]) -> None:
    """Upload a CSV, check its columns, score every row, offer the results as CSV."""
    st.caption(
        f"Upload a CSV with the {len(feature_names)} feature columns of this dataset "
        "(extra columns such as an id or label are kept). Every row is scored."
    )
    with st.expander("Required column names"):
        st.code(", ".join(feature_names))
        st.download_button(
            "Download an empty template",
            ",".join(feature_names) + "\n",
            file_name=f"{dataset_key}_template.csv",
            mime="text/csv",
            key="batch_tpl",
        )
    up = st.file_uploader("CSV", type=["csv"], key="csv_feat")
    if up is None:
        st.session_state.pop("batch_df", None)
        st.session_state["batch_ok"] = False
        return
    try:
        df = read_csv_bytes(up.getvalue())
        validate(df, feature_names)
    except BatchError as e:
        st.session_state["batch_ok"] = False
        st.error(f"Cannot score this file: {e}")
        return
    st.session_state["batch_df"] = df
    st.session_state["batch_ok"] = True
    st.success(f"{len(df):,} rows · all {len(feature_names)} required columns present.")
    if st.button(f"Score all {len(df):,} rows", key="batch_run"):
        try:
            st.session_state["batch_result"] = (dataset_key, model_name, score_table(dataset_key, df, model_name))
        except BatchError as e:
            st.error(f"Cannot score this file: {e}")
    saved = st.session_state.get("batch_result")
    if saved and saved[0] == dataset_key and saved[1] == model_name:
        scored = saved[2]
        s = summary(scored)
        c = s["counts"]
        line = (f"PD side {c.get('PD side', 0)} · Healthy side {c.get('Healthy side', 0)} · "
                f"Uncertain {c.get('Uncertain', 0)}")
        if "agreement_on_called_rows" in s:
            line += (f" · matches the file's labels on {s['agreement_on_called_rows']:.0%} "
                     f"of {s['called_rows']} called rows")
        st.markdown(f"**Results** · {line}")
        st.dataframe(scored, width="stretch", hide_index=True)
        st.download_button(
            "Download results CSV",
            scored.to_csv(index=False).encode("utf-8"),
            file_name=f"voicesense_{dataset_key}_{model_name}_scores.csv",
            mime="text/csv",
            key="batch_dl",
        )
