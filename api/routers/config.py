"""
Configuration management endpoints.
"""
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from api.config_manager import config_manager, RuntimeConfig


router = APIRouter()


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration. All fields are optional - only provide fields you want to update."""
    SAVE_INTERVAL: Optional[int] = Field(None, ge=1, le=1000, description="Interval for saving progress (1-1000)")
    TEMPERATURE: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature for generation (0.0-2.0)")
    NUM_WORKERS: Optional[int] = Field(None, ge=1, le=32, description="Number of parallel workers (1-32)")
    MAX_QUESTION_PER_CHUNK: Optional[int] = Field(None, ge=1, le=20, description="Max questions per chunk for single-hop (1-20)")
    MAX_QUESTION_GENERATED: Optional[int] = Field(None, ge=1, le=10000, description="Max total questions for single-hop (1-10000)")
    FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES: Optional[int] = Field(None, ge=1, le=10000, description="Max final answers to generate (1-10000)")
    PROPOSE_GENERATOR_MAX_QUESTIONS_PER_ENTITY: Optional[int] = Field(None, ge=1, le=20, description="Max questions per entity for multi-hop (1-20)")
    ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold for entity elimination (0.0-1.0)")


class ConfigResponse(BaseModel):
    """Response model for configuration"""
    config: RuntimeConfig
    message: str


class ConfigUpdateResponse(BaseModel):
    """Response model for configuration updates"""
    previous: RuntimeConfig
    current: RuntimeConfig
    updated_fields: list[str]
    message: str


@router.get("/", response_model=ConfigResponse)
async def get_config():
    """
    Get current runtime configuration.

    Returns all configurable parameters with their current values.
    These values will be used for NEW tasks. Running tasks are not affected.
    """
    current_config = config_manager.get_config()
    return ConfigResponse(
        config=current_config,
        message="Current runtime configuration (applies to new tasks only)"
    )


@router.put("/", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest):
    """
    Update runtime configuration for NEW tasks.

    **Important**: Changes only affect NEW tasks submitted after this update.
    Running tasks will continue with their original configuration.

    Only provided fields will be updated. All fields are optional.

    Parameters:
    - **SAVE_INTERVAL**: Interval for saving progress (1-1000)
    - **TEMPERATURE**: LLM temperature for generation (0.0-2.0)
    - **NUM_WORKERS**: Number of parallel workers (1-32)
    - **MAX_QUESTION_PER_CHUNK**: Max questions per chunk for single-hop (1-20)
    - **MAX_QUESTION_GENERATED**: Max total questions for single-hop (1-10000)
    - **FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES**: Max final answers to generate (1-10000)
    - **PROPOSE_GENERATOR_MAX_QUESTIONS_PER_ENTITY**: Max questions per entity for multi-hop (1-20)
    - **ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD**: Similarity threshold for entity elimination (0.0-1.0)

    Returns:
        Previous and current configuration with list of updated fields
    """
    # Get current config before updating
    previous_config = config_manager.get_config()

    # Filter out None values (only update provided fields)
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        return ConfigUpdateResponse(
            previous=previous_config,
            current=previous_config,
            updated_fields=[],
            message="No updates provided"
        )

    # Apply updates
    updated_config = config_manager.update_config(updates)

    return ConfigUpdateResponse(
        previous=previous_config,
        current=updated_config,
        updated_fields=list(updates.keys()),
        message=f"Configuration updated. Changes will apply to NEW tasks only."
    )


@router.post("/reset", response_model=ConfigResponse)
async def reset_config():
    """
    Reset configuration to default values.

    This clears all PUT updates and resets all parameters to their model defaults:
    - SAVE_INTERVAL: 10
    - TEMPERATURE: 0.6
    - NUM_WORKERS: 4
    - MAX_QUESTION_PER_CHUNK: 3
    - MAX_QUESTION_GENERATED: 300
    - FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES: 300
    - PROPOSE_GENERATOR_MAX_QUESTIONS_PER_ENTITY: 3
    - ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD: 0.65

    Affects NEW tasks only - running tasks are not affected.
    """
    config_manager.reset_to_defaults()
    current_config = config_manager.get_config()

    return ConfigResponse(
        config=current_config,
        message="Configuration reset to default values (applies to new tasks)"
    )
