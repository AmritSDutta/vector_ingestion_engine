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
# Query (POST /api/query/analyse or /api/query/hybrid_analyse)
# ---------------------------------------------------------
st.subheader("🔍 Query")

# Endpoint selection
endpoint_choice = st.selectbox(
    "Search Type",
    ["Semantic Search", "Hybrid Search"],
    index=0,
    help="Semantic: dense vector search | Hybrid: dense + sparse with RRF"
)

# Query input (text area instead of text input)
query = st.text_area(
    "Enter your query",
    placeholder="e.g., Python developer with machine learning experience...",
    height=150,
    max_chars=500  # Matches backend validation
)

# top_k selector
top_k = st.number_input("Top-K", min_value=1, value=3, max_value=10, step=1)

# Run query button
if st.button("🧙‍♂️ Run Query") and query:
    # Determine endpoint
    endpoint = "/api/query/analyse" if endpoint_choice == "Semantic Search" else "/api/query/hybrid_analyse"

    payload = {"q": query, "top_k": int(top_k)}

    with st.spinner(f"Running {endpoint_choice}…"):
        try:
            resp = requests.post(
                f"{API_BASE}{endpoint}",
                json=payload,
                timeout=180,
            )

            if resp.status_code == 200:
                data = resp.json()

                # Display results as formatted list
                st.subheader(f"📊 Results ({len(data.get('results', []))} found)")

                for idx, result in enumerate(data.get('results', []), 1):
                    with st.expander(f"Result #{idx} - Score: {result.get('final_score', 0):.3f}", expanded=False):
                        # Scores
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Final Score", f"{result.get('final_score', 0):.3f}")
                        with col2:
                            st.metric("Dense Score", f"{result.get('dense_score', 0):.3f}")
                        with col3:
                            st.metric("Rerank Score", f"{result.get('rerank_score', 0):.3f}")

                        # Payload
                        payload_data = result.get('payload', {})
                        if 'Name' in payload_data:
                            st.markdown(f"**Name:** {payload_data['Name']}")
                        if 'Category' in payload_data:
                            st.markdown(f"**Category:** {payload_data['Category']}")
                        if 'Skills' in payload_data:
                            skills = payload_data['Skills']
                            if isinstance(skills, list):
                                st.markdown(f"**Skills:** {', '.join(skills)}")
                            else:
                                st.markdown(f"**Skills:** {skills}")
                        if 'Summary' in payload_data:
                            st.markdown(f"**Summary:** {payload_data['Summary']}")
                        if 'Education' in payload_data:
                            st.markdown(f"**Education:** {payload_data['Education']}")

                        # Show raw payload in expander
                        with st.expander("Raw Payload"):
                            st.json(payload_data)
            else:
                st.error(f"Error {resp.status_code}")
                st.text(resp.text)
        except requests.exceptions.ReadTimeout:
            st.error("Request timed out — try a longer timeout.")
        except Exception as e:
            st.error(f"Request failed: {e}")
