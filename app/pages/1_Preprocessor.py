# ============================================================================
# Imports
# ============================================================================
import streamlit as st
import sys
import time
import json
import os
from pathlib import Path
from st_copy import copy_button
from dotenv import load_dotenv

# ============================================================================
# Setup
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv("app/demo_single_hop.env")

from src.routers.preprocessor import Preprocessor
from src import PROJECT_ROOT

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(layout="wide")

# ============================================================================
# Title and Introduction
# ============================================================================
st.title("Preprocessor")
st.markdown("### Introduction:")

intro_text = '''
1. Preprocess with LLM to add missing punctuations, removes extra spaces or weird characters.
2. Chunk by Sentence, each question would be a chunk
3. Add sentence level trackers e.g. [sen x] at the end of each sentence
'''
st.markdown(intro_text)

# ============================================================================
# View Prompts Section
# ============================================================================
with st.expander("View Prompts"):
    prompt_path = os.getenv("PREPROCESSOR_PROMPT_PATH")
    if prompt_path:
        try:
            full_path = os.path.join(PROJECT_ROOT, prompt_path)
            with open(full_path, 'r') as f:
                prompt_content = f.read()
            st.code(prompt_content, language="text")
        except Exception as e:
            st.error(f"Cannot load prompt: {e}")
    else:
        st.warning("PREPROCESSOR_PROMPT_PATH not set")

# ============================================================================
# Input Section
# ============================================================================
st.markdown("### Input:")

# Tab selection for input method
tab1, tab2 = st.tabs(["📁 Upload PDF", "📝 Use Default PDF"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=['pdf'],
        help="Upload a PDF file to process"
    )

with tab2:
    st.info("Using default PDF from environment configuration")
    default_pdf_path = os.getenv("PREPROCESSOR_PDF_PATH")
    if default_pdf_path:
        st.code(f"Path: {default_pdf_path}", language="text")
    else:
        st.warning("PREPROCESSOR_PDF_PATH not set in environment")

# ============================================================================
# Process Button and Logic
# ============================================================================
if st.button("Run Preprocessor", type="primary", use_container_width=True):
    start_time = time.time()

    with st.spinner("Processing PDF..."):
        try:
            preprocessor = Preprocessor()

            # Use uploaded file if provided, otherwise use default
            if uploaded_file is not None:
                st.info(f"Processing uploaded file: {uploaded_file.name}")
                chunk_contents, prompt_tokens, completion_tokens, success_num, all_num = preprocessor.run(pdf=uploaded_file)
            else:
                st.info("Processing default PDF from environment")
                prompt_tokens, completion_tokens, success_num, all_num = preprocessor.run()
                # Load the output to display
                output_path = os.getenv("PREPROCESSOR_CHUNKED_OUTPUT_PATH")
                if output_path:
                    with open(os.path.join(PROJECT_ROOT, output_path), 'r') as f:
                        chunk_contents = json.load(f)
                else:
                    st.error("PREPROCESSOR_CHUNKED_OUTPUT_PATH not set")
                    st.stop()

            elapsed_time = time.time() - start_time

            st.session_state.result = {
                'chunk_contents': chunk_contents,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'elapsed_time': elapsed_time
            }
            st.success(f"✓ Processing complete! Created {len(chunk_contents)} chunks in {elapsed_time:.2f}s")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            import traceback
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

# ============================================================================
# Output Display Section
# ============================================================================
if 'result' in st.session_state and 'chunk_contents' in st.session_state.result:
    result = st.session_state.result

    st.markdown("### Output:")

    # Copy button
    output_json = json.dumps(result['chunk_contents'], indent=2)
    copy_button(output_json, tooltip="Copy to clipboard", copied_label="Copied!")

    # JSON viewer
    with st.expander("View as JSON", expanded=False):
        st.json(result['chunk_contents'])

    # ========================================================================
    # Pagination Section
    # ========================================================================
    chunk_contents = result['chunk_contents']
    if isinstance(chunk_contents, list) and len(chunk_contents) > 0:
        # Initialize pagination state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0

        st.markdown("### Browse Items:")

        # Display current item
        current_item = chunk_contents[st.session_state.current_index]
        st.markdown(f"**Item {st.session_state.current_index + 1} of {len(chunk_contents)}**")

        with st.expander("View Item Details", expanded=True):
            st.json(current_item)

        # Navigation buttons
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button(
                "Previous",
                disabled=(st.session_state.current_index == 0),
                use_container_width=True
            ):
                st.session_state.current_index -= 1
                st.rerun()

        with col2:
            if st.button(
                "Next",
                disabled=(st.session_state.current_index == len(chunk_contents) - 1),
                use_container_width=True
            ):
                st.session_state.current_index += 1
                st.rerun()

    # ========================================================================
    # Metrics Section
    # ========================================================================
    st.markdown("---")
    col1, col2 = st.columns(2)

    col1.metric("Total Token Usage", f"{result['prompt_tokens'] + result['completion_tokens']:,}")
    col2.metric("Elapsed Time", f"{result['elapsed_time']:.2f}s")
