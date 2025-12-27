"""
Single-hop pipeline endpoints.
"""
import sys
from pathlib import Path
from fastapi import APIRouter
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.preprocessor import Preprocessor
from src.components.fact_extractor import FactExtractor
from src.components.propose_generator import ProposeGenerator
from src.components.final_answer_generator import FinalAnswerGenerator
from src.components.answer_evaluator import AnswerEvaluator
from src.utils.file_utils import load_json
from api.models import SingleHopResponse

router = APIRouter()


@router.post("/run", response_model=SingleHopResponse)
async def run_single_hop():
    """Run single-hop pipeline."""
    load_dotenv('single_hop.env')

    # Stage 1: Preprocessor
    preprocessor = Preprocessor()
    preprocessor.run()
    data = load_json(preprocessor.PREPROCESSOR_CHUNKED_OUTPUT_PATH)

    # Stage 2: FactExtractor
    fact_extractor = FactExtractor()
    data, _, _, total_facts, _, _ = fact_extractor.run(inputs=data)

    # Stage 3: ProposeGenerator
    propose_generator = ProposeGenerator()
    data, _, _, _, _ = propose_generator.run(inputs=data)

    # Stage 4: FinalAnswerGenerator
    final_answer_generator = FinalAnswerGenerator()
    data, _, _, _, _ = final_answer_generator.run(inputs=data)

    # Stage 5: AnswerEvaluator
    answer_evaluator = AnswerEvaluator()
    data, _, _, _, _ = answer_evaluator.run(inputs=data)

    return {
        "status": "completed",
        "total_chunks": len(data),
        "total_facts": total_facts,
        "results": data
    }
