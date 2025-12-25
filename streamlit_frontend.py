import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000"

st.set_page_config("Chat with Documents (API)")
st.header("Chat with Documents using FastAPI Backend 💬")


if "file_id" not in st.session_state:
    st.session_state["file_id"] = None
if "file_ids" not in st.session_state:
    st.session_state["file_ids"] = []

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your Document", type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "csv", "db"])

    if st.button("Submit & Process") and uploaded_file:
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getbuffer())}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                file_id = response.json()["file_id"]
                st.session_state["file_id"] = file_id
                # Track all uploaded file_ids for multi-doc querying
                if file_id not in st.session_state["file_ids"]:
                    st.session_state["file_ids"].append(file_id)
                st.success(f"Uploaded! file_id: {file_id}")
            else:
                st.error(f"Upload failed: {response.text}")

    # Multi-document selection
    if st.session_state["file_ids"]:
        st.markdown("### Select files to query:")
        selected_files = st.multiselect(
            "Choose one or more files:",
            st.session_state["file_ids"],
            default=st.session_state["file_ids"][:1]
        )
        st.session_state["selected_file_ids"] = selected_files
    else:
        st.session_state["selected_file_ids"] = []

st.write("---")


user_question = st.text_input("Ask a Question from the Document(s)")
image_file = st.file_uploader("Or upload an image for OCR-based question (optional)", type=["jpg", "jpeg", "png"], key="img")

if st.button("Ask") and user_question and st.session_state["selected_file_ids"]:
    image_base64 = None
    if image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "question": user_question,
        "file_id": ",".join(st.session_state["selected_file_ids"]),
        "image_base64": image_base64
    }
    with st.spinner("Querying..."):
        response = requests.post(f"{API_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.write("**Context:**")
            st.code(data["context"])
            st.write("**Answer:**")
            st.success(data["answer"])
            st.write("**Sources:**")
            for meta in data.get("sources", []):
                st.write(f"{meta.get('icon','')} `{meta.get('filename','')}` | Chunk: {meta.get('chunk','')} | Type: {meta.get('filetype','')}")
        else:
            st.error(f"Query failed: {response.text}")
