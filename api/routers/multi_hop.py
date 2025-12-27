"""
Multi-hop pipeline endpoints.
"""
import sys
from pathlib import Path
from fastapi import APIRouter
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.models import MultiHopResponse

router = APIRouter()


@router.post("/run", response_model=MultiHopResponse)
async def run_multi_hop():
    """Run multi-hop pipeline."""
    load_dotenv('multi_hop.env')

    # TODO: Implement multi-hop pipeline
    return {
        "status": "not_implemented",
        "message": "Multi-hop pipeline not yet implemented"
    }
