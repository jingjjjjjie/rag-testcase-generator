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

from src.routers.fact_extractor import FactExtractor
from src import PROJECT_ROOT

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(layout="wide")

# ============================================================================
# Title and Introduction
# ============================================================================
st.title("Fact Extractor")
st.markdown("### Introduction:")

intro_text = '''
**Purpose:**
Extracts objective facts and their referenced sentence numbers.

**Notes:**
- A fact can contain multiple sentence references:
  - Use `[Sen xx]` if only one sentence is referenced
  - Use `[Sen xx, xx, ...]` if multiple sentences are referenced
- The system is prompted to **remove discussions on widely recognized universal values and common knowledge**.
'''
st.markdown(intro_text)

# ============================================================================
# Prompt Section
# ============================================================================
st.markdown("### Prompt:")

# Tab selection for prompt method
prompt_tab1, prompt_tab2 = st.tabs(["📁 Use Default Prompt", "📝 Paste Prompt"])

with prompt_tab1:
    st.info("Using default prompt from environment configuration")
    prompt_path = os.getenv("FACT_EXTRACTOR_PROMPT_PATH")
    if prompt_path:
        with st.expander("View Default Prompt"):
            try:
                full_path = os.path.join(PROJECT_ROOT, prompt_path)
                with open(full_path, 'r') as f:
                    default_prompt_content = f.read()
                st.code(default_prompt_content, language="text")
            except Exception as e:
                st.error(f"Cannot load prompt: {e}")
    else:
        st.warning("FACT_EXTRACTOR_PROMPT_PATH not set")

with prompt_tab2:
    custom_prompt = st.text_area(
        'Prompt',
        height=200,
        placeholder='Paste your custom prompt here...',
        label_visibility="collapsed",
        key="custom_prompt_text"
    )

# ============================================================================
# Input Section
# ============================================================================
st.markdown("### Input:")

# Tab selection for input method
tab1, tab2 = st.tabs(["📁 Use Default Input","📝 Paste JSON"])

with tab1:
    st.info("Using default input from environment configuration")
    default_input_path = os.getenv("FACT_EXTRACTOR_INPUT_PATH")
    if default_input_path:
        st.code(f"Path: {default_input_path}", language="text")
    else:
        st.warning("FACT_EXTRACTOR_INPUT_PATH not set in environment")

with tab2:
    input_text = st.text_area(
        'Input',
        height=200,
        placeholder='Paste JSON output from preprocessor here...',
        label_visibility="collapsed",
        key="fact_input_text"
    )

# ============================================================================
# Process Button and Logic
# ============================================================================
if st.button("Run Fact Extractor", type="primary", use_container_width=True):
    start_time = time.time()

    with st.spinner("Processing..."):
        try:
            fact_extractor = FactExtractor()

            # Check if custom prompt was provided
            has_custom_prompt = 'custom_prompt_text' in st.session_state and st.session_state.custom_prompt_text

            # Determine if we're in direct mode
            direct_mode = input_text or has_custom_prompt

            # Use pasted JSON/prompt if provided, otherwise use default
            if direct_mode:
                if input_text:
                    st.info("Processing pasted JSON input")
                    json_inputs = json.loads(input_text)
                else:
                    st.info("Processing default input from environment")
                    # Load default input for direct mode
                    input_path = os.getenv("FACT_EXTRACTOR_INPUT_PATH")
                    if not input_path:
                        st.error("FACT_EXTRACTOR_INPUT_PATH not set")
                        st.stop()
                    with open(os.path.join(PROJECT_ROOT, input_path), 'r') as f:
                        json_inputs = json.load(f)

                if has_custom_prompt:
                    st.info("Using custom prompt")

                # Run in direct mode
                results, prompt_tokens, completion_tokens, success_num, all_num = fact_extractor.run(
                    inputs=json_inputs,
                    prompt=st.session_state.custom_prompt_text if has_custom_prompt else None
                )
            else:
                st.info("Processing default input and prompt from environment")
                prompt_tokens, completion_tokens, success_num, all_num = fact_extractor.run()
                # Load the output to display
                output_path = os.getenv("FACT_EXTRACTOR_OUTPUT_PATH")
                if output_path:
                    with open(os.path.join(PROJECT_ROOT, output_path), 'r') as f:
                        results = json.load(f)
                else:
                    st.error("FACT_EXTRACTOR_OUTPUT_PATH not set")
                    st.stop()

            elapsed_time = time.time() - start_time

            st.session_state.result = {
                'results': results,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'elapsed_time': elapsed_time
            }
            st.success(f"✓ Processing complete! Processed {len(results)} items in {elapsed_time:.2f}s")

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            import traceback
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Error during processing: {e}")
            import traceback
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

# ============================================================================
# Output Display Section
# ============================================================================
if 'result' in st.session_state and 'results' in st.session_state.result:
    result = st.session_state.result

    st.markdown("### Output:")

    # Copy button
    output_json = json.dumps(result['results'], indent=2)
    copy_button(output_json, tooltip="Copy to clipboard", copied_label="Copied!")

    # JSON viewer
    with st.expander("View as JSON", expanded=False):
        st.json(result['results'])

    # ========================================================================
    # Pagination Section
    # ========================================================================
    results = result['results']
    if isinstance(results, list) and len(results) > 0:
        # Initialize pagination state
        if 'fact_current_index' not in st.session_state:
            st.session_state.fact_current_index = 0

        st.markdown("### Browse Items:")

        # Display current item
        current_item = results[st.session_state.fact_current_index]
        st.markdown(f"**Item {st.session_state.fact_current_index + 1} of {len(results)}**")

        with st.expander("View Item Details", expanded=True):
            st.json(current_item)

        # Navigation buttons
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button(
                "Previous",
                disabled=(st.session_state.fact_current_index == 0),
                use_container_width=True,
                key="fact_prev"
            ):
                st.session_state.fact_current_index -= 1
                st.rerun()

        with col2:
            if st.button(
                "Next",
                disabled=(st.session_state.fact_current_index == len(results) - 1),
                use_container_width=True,
                key="fact_next"
            ):
                st.session_state.fact_current_index += 1
                st.rerun()

    # ========================================================================
    # Metrics Section
    # ========================================================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Questions/Chunks", f"{len(result['results']):,}")
    col2.metric("Total Token Usage", f"{result['prompt_tokens'] + result['completion_tokens']:,}")
    col3.metric("Elapsed Time", f"{result['elapsed_time']:.2f}s")
