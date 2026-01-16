import re
import os
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.api_utils import call_api_qwen
from src.utils.file_utils import save_json, read_text_file, extract_text_from_pdf
from src.utils.logger import info, error
from src import PROJECT_ROOT


class GenericPreprocessor:
    """Generic preprocessor using LangChain chunking. Works with any document format."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ";", " "]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

        self.PREPROCESSOR_PDF_PATH = None
        self.PREPROCESSOR_PROMPT_PATH = None
        self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = None

        if os.getenv("PREPROCESSOR_PDF_PATH", None) is not None:
            self.PREPROCESSOR_PDF_PATH = os.path.join(PROJECT_ROOT, os.getenv("PREPROCESSOR_PDF_PATH"))
            self.PREPROCESSOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PREPROCESSOR_PROMPT_PATH"))
            self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PREPROCESSOR_CHUNKED_OUTPUT_PATH"))

        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

    def process_chunk_text(self, chunk, counter):
        """Process chunk text and add sentence labels."""
        text = chunk.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', text)

        labeled = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if sent[-1] in ".!?":
                labeled.append(f"{sent[:-1]} [Sen {counter}]{sent[-1]}")
            else:
                labeled.append(f"{sent} [Sen {counter}]")
            counter += 1

        return " ".join(labeled), counter

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using LangChain."""
        return self.text_splitter.split_text(text)

    def process_chunks(self, chunks: List[str], doc_id: str = "doc1") -> List[Dict]:
        """Process chunks and add sentence labels."""
        chunk_contents = []
        sentence_counter = 1

        for i, chunk in enumerate(chunks):
            labeled_text, sentence_counter = self.process_chunk_text(chunk, sentence_counter)

            chunk_dict = {
                "id": f"{doc_id}_chunk{i + 1}",
                "origin_context": chunk,
                "context": labeled_text,
                "chunk_index": i,
                "char_count": len(chunk)
            }
            chunk_contents.append(chunk_dict)

        return chunk_contents

    def run(self, pdf: Optional[Union[str, object]] = None, text: Optional[str] = None, clean_with_llm: bool = True) -> tuple:
        """Run preprocessor. Provide pdf OR text. If both, text takes priority. Returns (chunks, prompt_tokens, completion_tokens, success, total)."""
        info("=" * 100)
        info("Running Generic Preprocessor (LangChain Chunking)".center(100))
        info("=" * 100)

        direct_mode = pdf is not None or text is not None
        total_prompt_tokens = 0
        total_completion_tokens = 0
        success_num = 0
        all_num = 2 if clean_with_llm else 1

        # Step 1: Get text
        try:
            if text is not None:
                raw_text = text
                info(f"Step 1/{all_num}: Using provided text ({len(raw_text)} characters)")
            elif pdf is not None:
                info(f"Step 1/{all_num}: Extracting text from provided PDF")
                raw_text = extract_text_from_pdf(pdf)
                info(f"Extracted {len(raw_text)} characters from PDF")
            else:
                if self.PREPROCESSOR_PDF_PATH is None:
                    raise EnvironmentError("No PDF/text provided and PREPROCESSOR_PDF_PATH not set")
                info(f"Step 1/{all_num}: Extracting text from PDF at {self.PREPROCESSOR_PDF_PATH}")
                raw_text = extract_text_from_pdf(self.PREPROCESSOR_PDF_PATH)
                info(f"Extracted {len(raw_text)} characters from PDF")

        except Exception as e:
            error(f"Error extracting PDF: {e}")
            if direct_mode:
                return [], total_prompt_tokens, total_completion_tokens, success_num, all_num
            return total_prompt_tokens, total_completion_tokens, success_num, all_num

        # Step 2: Clean text with LLM (optional)
        if clean_with_llm and self.PREPROCESSOR_PROMPT_PATH:
            try:
                info(f"Step 2/{all_num}: Cleaning text with LLM")
                data_cleaning_prompt = read_text_file(self.PREPROCESSOR_PROMPT_PATH)
                cleaned_text, prompt_tokens, completion_tokens = call_api_qwen(
                    data_cleaning_prompt + raw_text,
                    temperature=self.TEMPERATURE
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                success_num += 1
                info(f"Text cleaned successfully. Tokens Used: {prompt_tokens + completion_tokens}")
            except Exception as e:
                error(f"Error cleaning text with LLM: {e}, using raw text instead")
                cleaned_text = raw_text
        else:
            cleaned_text = raw_text
            if not clean_with_llm:
                info("Skipping LLM cleaning step")

        # Step 3: Chunk and label
        try:
            info(f"Step {all_num}/{all_num}: Chunking text with LangChain and adding sentence labels")
            chunks = self.chunk_text(cleaned_text)
            info(f"Created {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")

            chunk_contents = self.process_chunks(chunks)

            if not direct_mode and self.PREPROCESSOR_CHUNKED_OUTPUT_PATH:
                save_json(chunk_contents, self.PREPROCESSOR_CHUNKED_OUTPUT_PATH)

            success_num += 1
            info(f"Chunking completed successfully. Created {len(chunk_contents)} labeled chunks")

        except Exception as e:
            error(f"Error chunking text: {e}")
            if direct_mode:
                return [], total_prompt_tokens, total_completion_tokens, success_num, all_num
            return total_prompt_tokens, total_completion_tokens, success_num, all_num

        info(f"Total prompt tokens: {total_prompt_tokens}")
        info(f"Total completion tokens: {total_completion_tokens}")
        info(f"Task Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%")

        if direct_mode:
            return chunk_contents, total_prompt_tokens, total_completion_tokens, success_num, all_num
        return total_prompt_tokens, total_completion_tokens, success_num, all_num


if __name__ == "__main__":
    # 测试长文本
    load_dotenv("single_hop.env")
    text = open("tests/sample_long_text.txt").read()
    print(f"原始文本长度: {len(text)} 字符")

    preprocessor = GenericPreprocessor(chunk_size=800, chunk_overlap=100)
    chunks, prompt_tokens, completion_tokens, success_num, all_num = preprocessor.run(text=text, clean_with_llm=True)

    print(f"\n生成了 {len(chunks)} 个 chunks")
    print("=" * 50)
    print(chunks)
    for chunk in chunks:
        print(f"\n[{chunk['id']}] - {chunk['char_count']} chars")
        print("-" * 40)
        print(chunk['context'][:300] + "..." if len(chunk['context']) > 300 else chunk['context'])
        print()
