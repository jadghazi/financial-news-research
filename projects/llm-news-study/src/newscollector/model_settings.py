"""Local credential configuration and a metadata-only provider connection check.

Prediction requests are built in gemma.py; this panel never sends headlines.
"""
from __future__ import annotations

import json
import streamlit as st

from .core import now
from .gemma import MODEL, GemmaClient, credential, save_credential


def model_settings(db):
    try:
        credential(db.home)
        configured = True
    except (ValueError, OSError):
        configured = False
    meta_path = db.home / "google-model-check.json"
    checked = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    st.caption(f"Gemma 4 31B · {'Key saved locally' if configured else 'Key needed'} · No tools/search configured")
    with st.expander("Model settings & connection"):
        st.code(MODEL, language=None)
        st.write("The model receives only the ticker, target date and dated headlines. No opening/closing prices or scoring labels are sent.")
        with st.form("exp_key_form", clear_on_submit=True):
            key = st.text_input("Google API key", type="password", help="Saved locally in ignored data/google-credentials.json. Never included in dataset downloads.")
            save = st.form_submit_button("Save key")
        if save:
            try:
                save_credential(db.home, key)
                meta_path.unlink(missing_ok=True)
                checked = {}
                configured = True
                st.success("Key saved. You can check the connection now.")
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Check connection", disabled=not configured):
            client = None
            try:
                client = GemmaClient(credential(db.home))
                metadata = client.check()
                checked = {"checked_at": now(), "model": metadata}
                meta_path.write_text(json.dumps(checked), encoding="utf-8")
                st.success("Google accepted the key and returned the Gemma 4 31B model record.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            finally:
                if client:
                    client.close()
        if checked:
            st.caption(f"Last connection check: {checked['checked_at']} · model record version {checked['model'].get('version', 'unspecified')}")
        st.caption("Screenshot limits: 30 requests/min, 16K input tokens/min, 14.4K requests/day. This app uses lower safety budgets and pauses on quota errors. Only Gemma 4 31B is used; it never switches models automatically.")
        st.caption("Hosted requests need internet to reach Google. No browsing/tools are included in requests; this does not erase knowledge in model weights. Historical text versions may be unverified. Google free-tier data may be used to improve its products.")
