"""
Centralized configuration manager for runtime settings.
"""
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import threading


class RuntimeConfig(BaseModel):
    """Runtime configuration model"""
    SAVE_INTERVAL: int = Field(default=10, ge=1, le=1000, description="Interval for saving progress")
    TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0, description="LLM temperature for generation")
    NUM_WORKERS: int = Field(default=4, ge=1, le=32, description="Number of parallel workers")
    MAX_QUESTION_PER_CHUNK: int = Field(default=3, ge=1, le=20, description="Max questions per chunk (single-hop)")
    MAX_QUESTION_GENERATED: int = Field(default=300, ge=1, le=10000, description="Max total questions to generate (single-hop)")
    FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES: int = Field(default=300, ge=1, le=10000, description="Max final answers to generate")
    PROPOSE_GENERATOR_MAX_QUESTIONS_PER_ENTITY: int = Field(default=3, ge=1, le=20, description="Max questions per entity (multi-hop)")
    ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0, description="Similarity threshold for entity elimination (multi-hop)")


class ConfigManager:
    """Thread-safe configuration manager"""

    def __init__(self):
        self._config: RuntimeConfig = RuntimeConfig()
        self._lock = threading.Lock()
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables"""
        with self._lock:
            env_values = {}

            # Load each config field from environment
            for field_name in RuntimeConfig.model_fields.keys():
                env_value = os.getenv(field_name)
                if env_value is not None:
                    # Convert to appropriate type
                    field_info = RuntimeConfig.model_fields[field_name]
                    if field_info.annotation == int:
                        env_values[field_name] = int(env_value)
                    elif field_info.annotation == float:
                        env_values[field_name] = float(env_value)
                    else:
                        env_values[field_name] = env_value

            # Update config with environment values
            if env_values:
                self._config = RuntimeConfig(**env_values)

    def get_config(self) -> RuntimeConfig:
        """Get current configuration (thread-safe)"""
        with self._lock:
            return self._config.model_copy()

    def update_config(self, updates: Dict[str, Any]) -> RuntimeConfig:
        """Update configuration with new values (thread-safe)"""
        with self._lock:
            current_dict = self._config.model_dump()
            current_dict.update(updates)
            self._config = RuntimeConfig(**current_dict)

            # Update environment variables so components pick up new values
            for key, value in updates.items():
                os.environ[key] = str(value)

            return self._config.model_copy()

    def reload_from_env(self):
        """Reload configuration from environment variables"""
        self._load_from_env()

    def reload_from_file(self, env_file: str):
        """Reload configuration from a specific .env file"""
        from dotenv import load_dotenv
        import os

        # Clear existing env vars for config fields
        with self._lock:
            for field_name in RuntimeConfig.model_fields.keys():
                if field_name in os.environ:
                    del os.environ[field_name]

        # Load from specified file
        load_dotenv(env_file, override=True)

        # Reload config from newly loaded env vars
        self._load_from_env()

    def reset_to_defaults(self):
        """Reset all config fields to their default values from RuntimeConfig model"""
        with self._lock:
            # Clear all config-related environment variables
            for field_name in RuntimeConfig.model_fields.keys():
                if field_name in os.environ:
                    del os.environ[field_name]

            # Reset to model defaults
            self._config = RuntimeConfig()

            # Update environment variables with defaults
            for field_name, value in self._config.model_dump().items():
                os.environ[field_name] = str(value)

    def get_value(self, key: str) -> Any:
        """Get a specific configuration value"""
        with self._lock:
            return getattr(self._config, key, None)

    def set_value(self, key: str, value: Any):
        """Set a specific configuration value"""
        self.update_config({key: value})


# Global config manager instance
config_manager = ConfigManager()
