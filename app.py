import streamlit as st

from app.ui import render_app
from runtime.prewarm import prewarm_rag_runtime


def main() -> None:
    # Streamlit server is already listening when this script runs,
    # so Cloud Run health/startup checks can succeed while the
    # heavy RAG runtime is initialized before the demo controls
    # are exposed to users.
    try:
        with st.spinner(
            "Initializing telecom RAG runtime — loading FAISS, "
            "metadata and BGE-M3. First cold start may take a few minutes..."
        ):
            state = prewarm_rag_runtime()

        st.session_state["_rag_prewarm_state"] = state

    except Exception as exc:
        st.error(
            "The telecom RAG runtime could not be initialized. "
            "Please retry after the deployment/runtime issue is resolved."
        )
        st.exception(exc)
        st.stop()

    render_app()


if __name__ == "__main__":
    main()
