import os
import re
from src import PROJECT_ROOT
from src.utils.file_utils import load_json, save_json
from src.utils.logger import info

class QuestionExtractor:
    """
    Extracts final questions and answers from the pipeline output.

    This component processes the output from AnswerEvaluator and extracts
    only the valid questions with their corrected answers and citations.
    """

    def __init__(self):
        self.QUESTION_EXTRACTOR_INPUT_PATH = None
        self.QUESTION_EXTRACTOR_OUTPUT_PATH = None

        if os.getenv("QUESTION_EXTRACTOR_INPUT_PATH", None) is not None:
            self.QUESTION_EXTRACTOR_INPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("QUESTION_EXTRACTOR_INPUT_PATH"))
            self.QUESTION_EXTRACTOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("QUESTION_EXTRACTOR_OUTPUT_PATH"))
        else:
            raise EnvironmentError("Environment variables are not configured properly.")

        # Pattern for extracting citations like [Doc ..., Sen ...]
        self.citation_pattern = re.compile(r"\[([^\]]+)\]")

    def extract_origins(self, text):
        """
        Extract all [Doc ..., Sen ...] citations from text.

        Args:
            text (str): Text containing citations

        Returns:
            list: List of citation strings
        """
        if not text:
            return []
        return self.citation_pattern.findall(text)

    def process_chunk(self, chunk):
        """
        Process a single chunk and extract valid questions.

        Args:
            chunk (dict): A chunk from the pipeline output

        Returns:
            list: List of extracted question-answer pairs
        """
        results = []
        chunk_id = chunk.get("id")

        proposed = chunk.get("proposed-questions", {})
        for _, qblock in proposed.items():
            question = qblock.get("question")

            corrected = qblock.get("corrected-answer", {})
            short = corrected.get("short-answer", {})

            short_answer = short.get("answer")
            reason = short.get("reason")

            # Only include questions with all required fields
            if not (question and short_answer and reason):
                continue

            origins = self.extract_origins(short_answer)

            results.append({
                "context": chunk_id,
                "question": question,
                "short_answer": short_answer,
                "reason": reason,
                "origin": origins
            })

        return results

    def run(self, inputs=None):
        """
        Extract questions from pipeline output.

        Args:
            inputs (list, optional): Input data from previous pipeline stage.
                                    If None, will load from file.

        Returns:
            In direct mode: (extracted_questions, total_questions)
            In file mode: total_questions
        """
        info("=" * 100)
        info("Running Question Extractor".center(100))
        info("=" * 100)

        # Determine if we're in direct mode (inputs provided)
        direct_mode = inputs is not None

        # Load inputs if not provided
        if inputs is None:
            inputs = load_json(self.QUESTION_EXTRACTOR_INPUT_PATH)
            info(f"Loaded {len(inputs)} chunks from {os.path.relpath(self.QUESTION_EXTRACTOR_INPUT_PATH, PROJECT_ROOT)}.")

        # Extract questions from all chunks
        extracted_questions = []
        for chunk in inputs:
            chunk_results = self.process_chunk(chunk)
            extracted_questions.extend(chunk_results)

        total_questions = len(extracted_questions)
        info(f"Extracted {total_questions} valid question-answer pairs from {len(inputs)} chunks.")

        # Only save if not in direct mode
        if not direct_mode:
            info(f"Saving extracted questions to {os.path.relpath(self.QUESTION_EXTRACTOR_OUTPUT_PATH, PROJECT_ROOT)}.")
            save_json(extracted_questions, self.QUESTION_EXTRACTOR_OUTPUT_PATH)

        info(f"Question extraction completed: {total_questions} questions extracted.")

        if direct_mode:
            return extracted_questions, total_questions
        else:
            return total_questions


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("single_hop.env")

    question_extractor = QuestionExtractor()
    question_extractor.run()
