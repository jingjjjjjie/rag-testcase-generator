import os
from dotenv import load_dotenv
import copy

from src.utils.file_utils import save_json, load_json
from src.utils.logger import info, error
from src import PROJECT_ROOT

load_dotenv()


class AddEntityId:

    def __init__(self):

        info("=" * 100)
        info("Running Add Entity ID".center(100))
        info("=" * 100)

        self.ADD_ENTITY_ID_INPUT_PATH = None
        self.ADD_ENTITY_ID_OUTPUT_PATH = None

        if os.getenv("ADD_ENTITY_ID_INPUT_PATH", None) != None:
            self.ADD_ENTITY_ID_INPUT_PATH = os.getenv("ADD_ENTITY_ID_INPUT_PATH")
            self.ADD_ENTITY_ID_OUTPUT_PATH = os.getenv("ADD_ENTITY_ID_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variables are not defined correctly")
        
        # mandatory parameters to be defined in .env file
        self.ADD_ENTITY_ID_INPUT_PATH = os.path.join(PROJECT_ROOT, self.ADD_ENTITY_ID_INPUT_PATH)
        self.ADD_ENTITY_ID_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.ADD_ENTITY_ID_OUTPUT_PATH)

    def run(self, inputs=None):
        # Determine if we're in direct mode (inputs provided)
        direct_mode = inputs is not None

        # Load inputs if not provided
        if inputs is None:
            inputs = load_json(self.ADD_ENTITY_ID_INPUT_PATH)
            info(f"Loaded {len(inputs)} entries from {os.path.relpath(self.ADD_ENTITY_ID_INPUT_PATH, PROJECT_ROOT)}.")
        else:
            info(f"Using provided inputs ({len(inputs)} items)")
            # Make a deep copy to avoid modifying the original input
            inputs = copy.deepcopy(inputs)

        # Initialize entity ID counter (global across all inputs)
        entity_id_beg = 0
        processed_count = 0

        # Process each input document
        for cur_input in inputs:
            # Filter and validate entities
            filtered_entities = []

            # Skip if current input has no entities
            if "entity" not in cur_input:
                continue

            # Filter entities to only include those with both name and description
            for entity in cur_input['entity']:
                if "entity_name" in entity and "entity_description" in entity:
                    filtered_entities.append(entity)

            # Create mapping from entity name to entity ID
            entity_name_to_id = {}

            # Assign unique IDs to each entity
            for cur_entity in filtered_entities:
                # Only assign new ID if entity doesn't already have one
                if "entity_id" not in cur_entity:
                    cur_entity["entity_id"] = entity_id_beg

                # Store entity name to ID mapping for relationship processing
                entity_name_to_id[cur_entity["entity_name"]] = cur_entity["entity_id"]

                # Increment global entity ID counter
                entity_id_beg += 1

            # Update input with filtered entities
            cur_input['entity'] = filtered_entities

            # Filter relationships to only include those with valid entities
            filter_relationships = []
            for cur_relationship in cur_input['relationship']:
                # Extract source and target entity names
                source_entity_name = cur_relationship['source_entity_name']
                target_entity_name = cur_relationship['target_entity_name']

                # Only keep relationships where both entities exist in our entity list
                if source_entity_name in entity_name_to_id and target_entity_name in entity_name_to_id:
                    filter_relationships.append(cur_relationship)

            # Add entity IDs to each relationship
            for cur_relationship in filter_relationships:
                # Get entity names from relationship
                source_entity_name = cur_relationship['source_entity_name']
                target_entity_name = cur_relationship['target_entity_name']

                # Add source and target entity IDs using the mapping
                cur_relationship['source_entity_id'] = entity_name_to_id[source_entity_name]
                cur_relationship['target_entity_id'] = entity_name_to_id[target_entity_name]

            # Update input with filtered relationships
            cur_input['relationship'] = filter_relationships
            processed_count += 1

        # Only save if not in direct mode
        if not direct_mode:
            info(f'Saving {processed_count} processed entries to {os.path.relpath(self.ADD_ENTITY_ID_OUTPUT_PATH, PROJECT_ROOT)}.')
            save_json(inputs, self.ADD_ENTITY_ID_OUTPUT_PATH)

        info(f"Completed: {processed_count} entries processed, {entity_id_beg} unique entity IDs assigned")

        if direct_mode:
            # In direct mode, return the processed inputs along with stats
            return inputs, processed_count, entity_id_beg
        else:
            # Return total entities processed and total IDs assigned
            return processed_count, entity_id_beg


if __name__ == "__main__":
    entity_id_adder = AddEntityId()
    entity_id_adder.run()