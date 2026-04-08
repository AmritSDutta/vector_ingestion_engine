import textwrap

import requests
import streamlit as st

# -----------------------------------------
# Professional Styling (smaller, cleaner)
# -----------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-size: 14px !important;
            font-family: "Inter", "Segoe UI", Roboto, sans-serif !important;
        }
        h1, h2, h3, h4 {
            font-weight: 600 !important;
        }
        .stButton button {
            font-size: 14px !important;
            font-family: "Inter", "Segoe UI", Roboto, sans-serif !important;
        }
        .stTextInput input, .stNumberInput input {
            font-size: 14px !important;
        }
        .stMarkdown {
            font-size: 14px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
.answer-md {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 14px;
    padding: 14px 18px;
    border-radius: 8px;
    border: 1px solid #ddd;
    line-height: 1.45;
    white-space: pre-wrap;   /* keeps formatting AND wraps naturally */
}
</style>
""", unsafe_allow_html=True)


API_BASE = "http://localhost:8000"

st.set_page_config(layout="wide", page_title="InsightScope")
st.title("🧠 InsightScope — RAG Client 🦙")

# ---------------------------------------------------------
# Health Check (GET /)
# ---------------------------------------------------------
with st.expander("🩺 API Checkup"):
    if st.button("Ping API"):
        try:
            resp = requests.get(f"{API_BASE}/", timeout=10)
            if resp.status_code == 200:
                st.success("API is healthy ✅")
                st.json(resp.json())
            else:
                st.error(f"Health check failed ({resp.status_code})")
                st.text(resp.text)
        except Exception as e:
            st.error(f"Health request failed: {e}")

# ---------------------------------------------------------
# Sidebar: Ingestion & Collections
# ---------------------------------------------------------
with st.sidebar:
    st.header("🛠️ Administration")

    uploaded_files = st.file_uploader(
        "📤 Upload PDF file",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False,  # single file expected
    )
    ingest = st.button("📄 Ingest Files")

    st.markdown("---")
    st.header("Collections")

    if st.button("📚 List Collections", key="list_collections"):
        try:
            resp = requests.get(f"{API_BASE}/api/ingest/collections", timeout=60)
            if resp.status_code == 200:
                st.success("📚 Collections:")
                st.json(resp.json())
            else:
                st.error(f"Error {resp.status_code}")
                st.text(resp.text)
        except Exception as e:
            st.error(f"List collections failed: {e}")

    collection_to_delete = st.text_input("Collection name to delete")
    if st.button("🚮 Delete Collection", key="delete_collection") and collection_to_delete:
        try:
            resp = requests.post(
                f"{API_BASE}/api/ingest/delete",
                params={"collection_name": collection_to_delete},
                timeout=60,
            )
            if resp.status_code == 200:
                st.success(f"Delete result: {resp.json()}")
            else:
                st.error(f"Error {resp.status_code}")
                st.text(resp.text)
        except Exception as e:
            st.error(f"Delete failed: {e}")

# ---------------------------------------------------------
# Ingestion (POST /api/ingest/upload)
# ---------------------------------------------------------
if ingest and uploaded_files:
    st.info("Uploading to ingestion service…")

    # Normalize uploader output to a list (handles single-file uploader)
    if isinstance(uploaded_files, list):
        files_list = uploaded_files
    else:
        files_list = [uploaded_files]

    # Build multipart payload correctly from UploadedFile objects
    files_payload = []
    for f in files_list:
        try:
            filename = getattr(f, "name", "uploaded_file")
            content = f.read()
            files_payload.append(("files", (filename, content, "application/octet-stream")))
        except Exception as e:
            st.error(f"Failed reading file {getattr(f, 'name', '<unknown>')}: {e}")

    try:
        resp = requests.post(
            f"{API_BASE}/api/ingest/upload",
            files=files_payload,
            timeout=300,
        )
        if resp.status_code == 200:
            st.success("Ingestion complete")
            st.json(resp.json())
        else:
            st.error(f"Error {resp.status_code}")
            st.text(resp.text)
    except Exception as e:
        st.error(f"Ingestion failed: {e}")

# ---------------------------------------------------------
# Query (POST /api/query/analyse)
# ---------------------------------------------------------
st.subheader("🔍 Query")
query = st.text_input("Enter your query")
top_k = st.number_input("Top-K", min_value=1, value=3, max_value=10, step=1)

if st.button("🧙‍♂️ Run Query") and query:
    payload = {"q": query, "top_k": int(top_k)}
    with st.spinner("Waiting for query results…"):
        try:
            resp = requests.post(
                f"{API_BASE}/api/query/analyse",
                json=payload,
                timeout=180,
            )
            if resp.status_code == 200:
                data = resp.json()

                # 1) Show only the "answer" section prominently
                if "answer" in data:
                    st.subheader("🧾 Answer")
                    st.markdown(f"<div class='answer-md'>{data['answer']}</div>", unsafe_allow_html=True)

                # 2) Show the whole raw response collapsed
                with st.expander("📦 Full Response (raw JSON)", expanded=False):
                    st.json(data)
            else:
                st.error(f"Error {resp.status_code}")
                st.text(resp.text)
        except requests.exceptions.ReadTimeout:
            st.error("Request timed out — try a longer timeout.")
        except Exception as e:
            st.error(f"Request failed: {e}")
