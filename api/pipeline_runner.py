"""
Pipeline Runner with Progress Tracking.

Executes the complete RAG testcase generation pipeline as a background task
with real-time progress updates.
"""
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('single_hop.env')

from api.task_manager import task_manager
from api.models import TaskStatus, StageInfo
from src.components.preprocessor import Preprocessor
from src.components.fact_extractor import FactExtractor
from src.components.propose_generator import ProposeGenerator
from src.components.final_answer_generator import FinalAnswerGenerator
from src.components.answer_evaluator import AnswerEvaluator
from src.components.extract_questions import QuestionExtractor
from src.utils.logger import info, error


def update_stage_progress(task_id: str, stage_name: str,
                         status: TaskStatus, progress: float,
                         items_processed: int = 0, items_total: int = 0,
                         tokens: int = 0, error_msg: str = None):
    """
    Helper function to update stage progress.

    Args:
        task_id: UUID of the task
        stage_name: Name of the pipeline stage
        status: Current status of the stage
        progress: Progress from 0.0 to 1.0
        items_processed: Number of items completed
        items_total: Total number of items
        tokens: Tokens used by this stage
        error_msg: Error message if any
    """
    stage_info = StageInfo(
        name=stage_name,
        status=status,
        progress=progress,
        items_processed=items_processed,
        items_total=items_total,
        tokens_used=tokens,
        start_time=datetime.now() if status == TaskStatus.RUNNING else None,
        end_time=datetime.now() if status in [TaskStatus.COMPLETED, TaskStatus.FAILED] else None,
        error=error_msg
    )
    task_manager.update_stage(task_id, stage_name, stage_info)


def run_pipeline_with_tracking(task_id: str):
    """
    Run the complete single-hop pipeline with progress tracking.
    Updates task status at each stage.

    This function follows the exact pattern from tester_single_hop.py
    and tracks all the statistics you want:
    - total_prompt_tokens
    - total_completion_tokens
    - total_chunks (len(data))
    - total_facts_extracted
    - total_questions_generated
    - total_extracted_questions

    Can be cancelled by calling task_manager.request_cancel(task_id).

    Args:
        task_id: UUID of the task to execute
    """
    try:
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        info(f"Starting pipeline execution for task {task_id}")

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # ========================================================================
        # Stage 1: Preprocessor - Load and chunk document
        # ========================================================================

        # Check cancellation before starting
        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled before starting")
            return

        update_stage_progress(task_id, "preprocessor", TaskStatus.RUNNING, 0.0)
        preprocessor = Preprocessor()
        preprocessor_result = preprocessor.run()

        # Preprocessor doesn't support direct mode, must load from file
        from src.utils.file_utils import load_json
        preprocessor_output_path = preprocessor.PREPROCESSOR_CHUNKED_OUTPUT_PATH
        data = load_json(preprocessor_output_path)

        prompt_tokens, completion_tokens, success_num, all_num = preprocessor_result
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        # mark as complete
        update_stage_progress(task_id, "preprocessor", TaskStatus.COMPLETED, 1.0,
                            items_processed=len(data), items_total=len(data),
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: Preprocessor completed. Generated {len(data)} chunks.")

        # ========================================================================
        # Stage 2: FactExtractor - Extract objective facts
        # ========================================================================

        # Check cancellation between stages
        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after preprocessor")
            return

        update_stage_progress(task_id, "fact_extractor", TaskStatus.RUNNING, 0.0)
        fact_extractor = FactExtractor()

        # Returns: inputs, total_prompt_tokens, total_completion_tokens, total_facts_extracted, success_rate
        data, prompt_tokens, completion_tokens, total_facts_extracted, success_rate = fact_extractor.run(inputs=data)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        update_stage_progress(task_id, "fact_extractor", TaskStatus.COMPLETED, 1.0,
                            items_processed=len(data), items_total=len(data),
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: FactExtractor completed. Extracted {total_facts_extracted} facts (success rate: {success_rate}).")

        # ========================================================================
        # Stage 3: ProposeGenerator - Generate proposed questions
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after fact_extractor")
            return

        update_stage_progress(task_id, "propose_generator", TaskStatus.RUNNING, 0.0)
        propose_generator = ProposeGenerator()

        # Returns: inputs, total_prompt_tokens, total_completion_tokens, success_num, all_num, total_questions_generated
        data, prompt_tokens, completion_tokens, success_num, all_num, total_questions_generated = propose_generator.run(inputs=data)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        update_stage_progress(task_id, "propose_generator", TaskStatus.COMPLETED, 1.0,
                            items_processed=success_num, items_total=all_num,
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: ProposeGenerator completed. Generated {total_questions_generated} questions from {success_num}/{all_num} chunks.")

        # ========================================================================
        # Stage 4: FinalAnswerGenerator - Generate final answers
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after propose_generator")
            return

        update_stage_progress(task_id, "final_answer_generator", TaskStatus.RUNNING, 0.0)
        final_answer_generator = FinalAnswerGenerator()

        # Returns: inputs, total_prompt_tokens, total_completion_tokens, success_num, all_num
        data, prompt_tokens, completion_tokens, success_num, all_num = final_answer_generator.run(inputs=data)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        update_stage_progress(task_id, "final_answer_generator", TaskStatus.COMPLETED, 1.0,
                            items_processed=success_num, items_total=all_num,
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: FinalAnswerGenerator completed. Evaluated {success_num}/{all_num} chunks.")

        # ========================================================================
        # Stage 5: AnswerEvaluator - Evaluate answers
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after final_answer_generator")
            return

        update_stage_progress(task_id, "answer_evaluator", TaskStatus.RUNNING, 0.0)
        answer_evaluator = AnswerEvaluator()

        # Returns: data, new_gen_num, all_num, total_prompt_tokens, total_completion_tokens
        data, new_gen_num, all_num, prompt_tokens, completion_tokens = answer_evaluator.run(inputs=data)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        update_stage_progress(task_id, "answer_evaluator", TaskStatus.COMPLETED, 1.0,
                            items_processed=new_gen_num, items_total=all_num,
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: AnswerEvaluator completed. Evaluated {new_gen_num}/{all_num} scoring tasks.")

        # ========================================================================
        # Stage 6: QuestionExtractor - Extract final valid questions
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after answer_evaluator")
            return

        update_stage_progress(task_id, "question_extractor", TaskStatus.RUNNING, 0.0)
        question_extractor = QuestionExtractor()

        # Returns: extracted_questions, total_questions
        extracted_questions, total_extracted_questions = question_extractor.run(inputs=data)

        update_stage_progress(task_id, "question_extractor", TaskStatus.COMPLETED, 1.0,
                            items_processed=total_extracted_questions,
                            items_total=total_extracted_questions)
        info(f"Task {task_id}: QuestionExtractor completed. Extracted {total_extracted_questions} valid questions.")

        # ========================================================================
        # Save final outputs 
        # ========================================================================

        from src import PROJECT_ROOT
        from src.utils.file_utils import save_json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full pipeline output
        full_output_path = os.path.join(
            PROJECT_ROOT,
            "runs",
            f"[{timestamp}]full_output.json"
        )
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        save_json(data, full_output_path)
        info(f"Full pipeline output saved to {full_output_path}")

        # Save extracted questions
        extracted_output_path = os.path.join(
            PROJECT_ROOT,
            "runs",
            f"[{timestamp}]extracted_questions.json"
        )
        save_json(extracted_questions, extracted_output_path)
        info(f"Extracted questions saved to {extracted_output_path}")

        # ========================================================================
        # Store final results - tracking exactly what tester_single_hop.py tracks
        # ========================================================================

        result = {
            "data": data,
            "extracted_questions": extracted_questions,
            "total_chunks": len(data),  # Total Chunks Processed
            "total_facts": total_facts_extracted,  # Total Facts Extracted
            "total_questions_generated": total_questions_generated,  # Total Questions Generated
            "total_valid_questions_extracted": total_extracted_questions,  # Total Valid Questions Extracted
            "total_prompt_tokens": total_prompt_tokens,  # Total Prompt Tokens
            "total_completion_tokens": total_completion_tokens,  # Total Completion Tokens
            "full_output_path": full_output_path,  # Path to saved full output
            "extracted_output_path": extracted_output_path  # Path to saved extracted questions
        }

        task_manager.store_result(task_id, result)
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

        # Log final statistics like tester_single_hop.py does
        info("=" * 100)
        info(f"Task {task_id}: PIPELINE COMPLETED SUCCESSFULLY".center(100))
        info("=" * 100)
        info(f"Total Prompt Tokens: {total_prompt_tokens}")
        info(f"Total Completion Tokens: {total_completion_tokens}")
        info(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
        info(f"Total Chunks Processed: {len(data)}")
        info(f"Total Questions Generated: {total_questions_generated}")
        info(f"Total Valid Questions Extracted: {total_extracted_questions}")
        info("=" * 100)

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
        error(f"Task {task_id} failed: {error_msg}")
        task_manager.set_error(task_id, error_msg)
