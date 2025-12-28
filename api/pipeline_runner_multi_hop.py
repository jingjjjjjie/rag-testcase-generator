"""
Pipeline Runner for Multi-Hop with Progress Tracking.

Executes the complete multi-hop RAG testcase generation pipeline as a background task
with real-time progress updates.
"""
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('multi_hop.env')

from api.task_manager import task_manager
from api.models import TaskStatus, StageInfo
from src.components.preprocessor import Preprocessor
from src.components.entity_extractor import EntityExtractor
from src.components.add_entity_id import AddEntityId
from src.components.entity_eliminator import EntityEliminator
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


def run_multi_hop_pipeline_with_tracking(task_id: str):
    """
    Run the complete multi-hop pipeline with progress tracking.
    Updates task status at each stage.

    This function follows the exact pattern from tester_multi_hop.py
    and tracks all the statistics:
    - total_prompt_tokens
    - total_completion_tokens
    - total_chunks (len(data))
    - total_entities
    - total_relationships
    - unique_count (entities after elimination)
    - total_questions_generated
    - total_extracted_questions

    Can be cancelled by calling task_manager.request_cancel(task_id).

    Args:
        task_id: UUID of the task to execute
    """
    try:
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        info(f"Starting multi-hop pipeline execution for task {task_id}")

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
        # Stage 2: EntityExtractor - Extract entities and relationships
        # ========================================================================

        # Check cancellation between stages
        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after preprocessor")
            return

        update_stage_progress(task_id, "entity_extractor", TaskStatus.RUNNING, 0.0)
        entity_extractor = EntityExtractor()

        # Returns: inputs, total_prompt_tokens, total_completion_tokens, success_rate, total_entities, total_relationships
        data, prompt_tokens, completion_tokens, success_rate, total_entities, total_relationships = entity_extractor.run(inputs=data)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        update_stage_progress(task_id, "entity_extractor", TaskStatus.COMPLETED, 1.0,
                            items_processed=len(data), items_total=len(data),
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: EntityExtractor completed. Extracted {total_entities} entities and {total_relationships} relationships (success rate: {success_rate}).")

        # ========================================================================
        # Stage 3: AddEntityId - Assign unique IDs to entities
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after entity_extractor")
            return

        update_stage_progress(task_id, "add_entity_id", TaskStatus.RUNNING, 0.0)
        add_entity_id = AddEntityId()

        # Returns: inputs, processed_count, entity_id_beg
        data, processed_count, entity_id_beg = add_entity_id.run(inputs=data)

        update_stage_progress(task_id, "add_entity_id", TaskStatus.COMPLETED, 1.0,
                            items_processed=processed_count, items_total=processed_count)
        info(f"Task {task_id}: AddEntityId completed. Processed {processed_count} chunks, assigned IDs starting from {entity_id_beg}.")

        # ========================================================================
        # Stage 4: EntityEliminator - Resolve duplicate entities
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after add_entity_id")
            return

        update_stage_progress(task_id, "entity_eliminator", TaskStatus.RUNNING, 0.0)
        entity_eliminator = EntityEliminator()

        # Returns: inputs, entityid2entityid, total_prompt_tokens, total_completion_tokens, original_count, unique_count
        data, entityid2entityid, prompt_tokens, completion_tokens, original_count, unique_count = entity_eliminator.run(inputs=data)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        task_manager.update_tokens(task_id, prompt_tokens, completion_tokens)

        update_stage_progress(task_id, "entity_eliminator", TaskStatus.COMPLETED, 1.0,
                            items_processed=unique_count, items_total=original_count,
                            tokens=prompt_tokens + completion_tokens)
        info(f"Task {task_id}: EntityEliminator completed. Reduced {original_count} entities to {unique_count} unique entities.")

        # ========================================================================
        # Stage 5: ProposeGenerator - Generate proposed questions
        # ========================================================================

        if task_manager.is_cancelled(task_id):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            info(f"Task {task_id} cancelled after entity_eliminator")
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
        info(f"Task {task_id}: ProposeGenerator completed. Generated {total_questions_generated} questions from {success_num}/{all_num} entities.")

        # ========================================================================
        # Stage 6: FinalAnswerGenerator - Generate final answers
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
        info(f"Task {task_id}: FinalAnswerGenerator completed. Evaluated {success_num}/{all_num} items.")

        # ========================================================================
        # Stage 7: AnswerEvaluator - Evaluate answers
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
        # Stage 8: QuestionExtractor - Extract final valid questions
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
            f"[{timestamp}]multi_hop_full_output.json"
        )
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        save_json(data, full_output_path)
        info(f"Full pipeline output saved to {full_output_path}")

        # Save extracted questions
        extracted_output_path = os.path.join(
            PROJECT_ROOT,
            "runs",
            f"[{timestamp}]multi_hop_extracted_questions.json"
        )
        save_json(extracted_questions, extracted_output_path)
        info(f"Extracted questions saved to {extracted_output_path}")

        # ========================================================================
        # Store final results - tracking exactly what tester_multi_hop.py tracks
        # ========================================================================

        result = {
            "data": data,
            "extracted_questions": extracted_questions,
            "total_chunks": len(data) if isinstance(data, list) else len(data.values()),
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "original_entity_count": original_count,
            "unique_entity_count": unique_count,
            "total_questions_generated": total_questions_generated,
            "total_valid_questions_extracted": total_extracted_questions,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "full_output_path": full_output_path,
            "extracted_output_path": extracted_output_path
        }

        task_manager.store_result(task_id, result)
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

        # Log final statistics like tester_multi_hop.py does
        info("=" * 100)
        info(f"Task {task_id}: MULTI-HOP PIPELINE COMPLETED SUCCESSFULLY".center(100))
        info("=" * 100)
        info(f"Total Prompt Tokens: {total_prompt_tokens}")
        info(f"Total Completion Tokens: {total_completion_tokens}")
        info(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
        info(f"Total Chunks Processed: {result['total_chunks']}")
        info(f"Total Entities Extracted: {total_entities}")
        info(f"Total Relationships Extracted: {total_relationships}")
        info(f"Unique Entities After Elimination: {unique_count}")
        info(f"Total Questions Generated: {total_questions_generated}")
        info(f"Total Valid Questions Extracted: {total_extracted_questions}")
        info("=" * 100)

    except Exception as e:
        error_msg = f"Multi-hop pipeline failed: {str(e)}\n{traceback.format_exc()}"
        error(f"Task {task_id} failed: {error_msg}")
        task_manager.set_error(task_id, error_msg)
