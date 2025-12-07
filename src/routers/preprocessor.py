import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.utils.api_utils import call_api_qwen
from src.utils.file_utils import save_json, load_json, read_text_file
from src.utils.string_utils import extract_text_from_pdf
from src.utils.logger import info, error
from src import PROJECT_ROOT

from src.routers import split_by_questions

class Preprocessor:
    def __init__(self):
        info("=" * 100)
        info("Running Preprocessor".center(100))
        info("=" * 100)
        self.PREPROCESSOR_PDF_PATH = None
        self.PREPROCESSOR_PROMPT_PATH = None
        self.PREPROCESSOR_CLEANED_OUTPUT_PATH = None
        self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = None
        self.PREPROCESSOR_CLEANED_PDF_PATH = None

        if os.getenv("PREPROCESSOR_PDF_PATH", None) != None:
            self.PREPROCESSOR_PDF_PATH = os.getenv("PREPROCESSOR_PDF_PATH")
            self.PREPROCESSOR_PROMPT_PATH = os.getenv("PREPROCESSOR_PROMPT_PATH")
            self.PREPROCESSOR_CLEANED_OUTPUT_PATH = os.getenv("PREPROCESSOR_CLEANED_OUTPUT_PATH")
            self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = os.getenv("PREPROCESSOR_CHUNKED_OUTPUT_PATH")
            self.PREPROCESSOR_CLEANED_PDF_PATH = os.getenv("PREPROCESSOR_CLEANED_PDF_PATH")
        else:
            raise EnvironmentError("Environment variables are not defined correctly")

        self.PREPROCESSOR_PDF_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_PDF_PATH)
        self.PREPROCESSOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_PROMPT_PATH)
        self.PREPROCESSOR_CLEANED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_CLEANED_OUTPUT_PATH)
        self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_CHUNKED_OUTPUT_PATH)
    
    def process_chunk_text(self, chunk, counter):
        """
        Process chunk text and add sentence labels
        """
        lines = []
        for line in chunk.split('\n'):
            sentences = re.split(r'(?<=[.!?])\s+', line)
            new_line = []

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                # Tag each sentence
                if sent[-1] in ".!?":
                    new_line.append(f"{sent[:-1]} [Sen {counter}]{sent[-1]}")
                else:
                    new_line.append(f"{sent} [Sen {counter}]")

                counter += 1

            lines.append(" ".join(new_line))

        return "\n".join(lines), counter

    def extract_question_from_chunk(self, question):
        """
        Extract the question text from a chunk
        """
        match = re.match(r"(\d+\))\s*(.*?)([?.])", question)
        if match:
            q_number = match.group(1)
            q_text = match.group(2).strip()
            punctuation = match.group(3)
            question_only = f"{q_number} {q_text}{punctuation}"
            answer_body = question[len(match.group(0)):].strip()
        else:
            question_only = ""
            answer_body = question

        return question_only, answer_body

    def process_questions(self, cleaned_questions):
        """
        Process questions and add sentence labels

        Returns:
            List of chunk_contents
        """
        chunk_contents = []
        counter = 1
        question_no = 1

        for question in cleaned_questions:
            question_only, answer_body = self.extract_question_from_chunk(question)
            q_processed, counter = self.process_chunk_text(answer_body, counter)

            chunk_dict = {
                "id": f"doc1_question{question_no}",
                "question": question_only,
                "origin_context": question,
                "context": q_processed
            }
            chunk_contents.append(chunk_dict)
            question_no += 1

        return chunk_contents

    def run(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0
        success_num = 0
        all_num = 3

        try:
            info(f"Step 1/3: Cleaning PDF at {self.PREPROCESSOR_PDF_PATH}")
            data_cleaning_prompt = read_text_file(self.PREPROCESSOR_PROMPT_PATH)
            pdf_text = extract_text_from_pdf(self.PREPROCESSOR_PDF_PATH)
            cleaned_pdf_text, prompt_tokens, completion_tokens = call_api_qwen(data_cleaning_prompt + pdf_text)
            save_json(cleaned_pdf_text, self.PREPROCESSOR_CLEANED_PDF_PATH)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            success_num += 1
            info(f"PDF cleaned successfully. Tokens Used: {prompt_tokens + completion_tokens}")
        except Exception as e:
            error(f"Error cleaning PDF: {e}")
            return total_prompt_tokens, total_completion_tokens, success_num, all_num

        try:
            info(f"Step 2/3: Splitting questions from cleaned PDF")
            question_set = split_by_questions(cleaned_pdf_text)
            save_json(question_set, self.PREPROCESSOR_CLEANED_OUTPUT_PATH)
            success_num += 1
            info(f"Questions split successfully. Found {len(question_set)} questions.")

        except Exception as e:
            error(f"Error splitting questions: {e}")
            return total_prompt_tokens, total_completion_tokens, success_num, all_num

        try:
            info(f"Step 3/3: Adding sentence labeling and chunking corpus")
            cleaned_questions = load_json(self.PREPROCESSOR_CLEANED_OUTPUT_PATH)
            chunk_contents = self.process_questions(cleaned_questions)
            save_json(chunk_contents, self.PREPROCESSOR_CHUNKED_OUTPUT_PATH)
            success_num += 1
            info(f"Questions chunked successfully. Created {len(chunk_contents)} chunks")
        except Exception as e:
            error(f"Error chunking questions or building corpus: {e}")
            return total_prompt_tokens, total_completion_tokens, success_num, all_num
        
        info(f"Total prompt tokens: {total_prompt_tokens}")
        info(f"Total completion tokens: {total_completion_tokens}")
        info(f"Task Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%")

        return total_prompt_tokens, total_completion_tokens, success_num, all_num


if __name__ == "__main__":
    load_dotenv('single_hop.env')
    preprocessor = Preprocessor()
    preprocessor.run()