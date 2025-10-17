import os, json, streamlit as st
from pipeline import answer_with_traces

st.set_page_config(page_title="CoT-RAG: Explainable Reasoning Search", layout="wide")
st.title("🧠 CoT-RAG: Explainable Reasoning Search")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("LLM backend", ["OLLAMA", "OPENAI"])
    os.environ["LLM_BACKEND"] = backend
    if backend == "OLLAMA":
        os.environ["OLLAMA_MODEL"] = st.text_input("Ollama model", "llama3")
        os.environ["OLLAMA_HOST"] = st.text_input("Ollama host", "http://localhost:11434")
    else:
        os.environ["OPENAI_MODEL"] = st.text_input("OpenAI model", "gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = st.text_input("OpenAI API Key", type="password")
    k = st.slider("Top-k passages", 1, 15, 6)
    n_samples = st.slider("CoT samples (self-consistency)", 1, 10, 5)
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

st.markdown("Add your `.txt`/`.md` documents under `docs/`, then ask a question.")

query = st.text_input("Your question")
if st.button("Run CoT-RAG") and query:
    with st.spinner("Running retrieval + CoT sampling..."):
        result = answer_with_traces(query, k=k, n_samples=n_samples, temperature=temp)

    st.subheader("✅ Final Answer")
    st.write(result["final_answer"])

    st.subheader("🔎 Retrieved Sources")
    for i, h in enumerate(result["retrieval"], 1):
        st.markdown(f"**{i}. {h['path']}**  \nscore: `{h['score']:.3f}`")
        with st.expander("View snippet"):
            st.code(h["snippet"])

    st.subheader("🧵 Reasoning Traces (CoT samples)")
    for i, s in enumerate(result["samples"], 1):
        st.markdown(f"**Sample {i}**")
        st.markdown("**Reasoning:**")
        st.write(s.get("reasoning","").strip())
        st.markdown("**Answer:**")
        st.info(s.get("final_answer","").strip())
