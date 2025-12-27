import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm

from src.utils.api_utils import call_api_qwen
from src.utils.file_utils import save_json, load_json, read_text_file
from src.utils.logger import info, error
from src import PROJECT_ROOT

class EntityExtractor:
    def __init__(self):

        self.ENTITY_EXTRACTOR_PROMPT_PATH = None
        self.ENTITY_EXTRACTOR_INPUT_PATH = None
        self.ENTITY_EXTRACTOR_OUTPUT_PATH = None
        self.TEMPERATURE = None
        self.NUM_WORKERS = None
        self.SAVE_INTERVAL = None

        if os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH", None) != None:
            self.ENTITY_EXTRACTOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH"))
            self.ENTITY_EXTRACTOR_INPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("ENTITY_EXTRACTOR_INPUT_PATH"))
            self.ENTITY_EXTRACTOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("ENTITY_EXTRACTOR_OUTPUT_PATH"))
        else:
            raise EnvironmentError("Environment variables are not defined correctly")

        # optional parameter with default values
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
        self.NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
        self.SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "10"))

        # Thread-safe counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_entities = 0
        self.total_relationships = 0
        self.token_lock = threading.Lock()

    def process_input(self, cur_input, entity_extractor_prompt, i):
        """Process a single input to extract entities and relationships"""
        try:
            context = cur_input['context']
            cur_entity_extractor_prompt = entity_extractor_prompt.replace('[[CONTEXT]]', context)
            entity_extractor_response, prompt_tokens, completion_tokens = call_api_qwen(cur_entity_extractor_prompt,temperature=self.TEMPERATURE)

            # Thread-safe token accumulation
            with self.token_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

            extract_entity = extract_entity_from_output(entity_extractor_response) # Parser function: Extract entities and relationships from model output with regex
            entities, relationships, is_complete = (
                extract_entity['entities'],
                extract_entity['relationships'],
                extract_entity['is_complete']
            )

            # Filter entities
            filtered_entities = []
            for entity in entities:
                # add the extracted entities to the list: filtered entities
                if "entity_name" in entity and "entity_description" in entity:
                    filtered_entities.append(entity)

            # Filter relationships to only include those with valid entities
            entity_names = [entity['entity_name'] for entity in filtered_entities]
            filtered_relationships = []
            for relationship in relationships:
                source_entity_name = relationship['source_entity_name']
                target_entity_name = relationship['target_entity_name']
                # iterates through relationships: get source name and target name from each relationships,
                # then validates: only appends relationships where both entities are present in entity_names
                if source_entity_name in entity_names and target_entity_name in entity_names:
                    filtered_relationships.append(relationship)

            # Thread-safe counts accumulation
            num_entities = len(filtered_entities)
            num_relationships = len(filtered_relationships)
            with self.token_lock:
                self.total_entities += num_entities
                self.total_relationships += num_relationships

            result = {
                **cur_input,
                'entity': filtered_entities,
                'relationship': filtered_relationships
            }
            return result, i
        except Exception as e:
            error(f"Error processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None

    def run(self, inputs=None):
        
        info("=" * 100)
        info("Running Entity Extractor".center(100))
        info("=" * 100)

        """Main entity extraction pipeline"""
        # Determine if we're in direct mode (inputs provided)
        direct_mode = inputs is not None

        if inputs is None: # Indirect Mode, output to a json file
            # determine if the entity extractor has been run
            if os.path.exists(self.ENTITY_EXTRACTOR_OUTPUT_PATH):
                inputs = load_json(self.ENTITY_EXTRACTOR_OUTPUT_PATH)
                info(f"Resuming from existing output: {len(inputs)} entries loaded")
            else:
                inputs = load_json(self.ENTITY_EXTRACTOR_INPUT_PATH)
                info(f"Loaded {len(inputs)} inputs from {self.ENTITY_EXTRACTOR_INPUT_PATH}")

        else: # direct Mode, return a json object
            pass  # Direct mode - inputs provided

        entity_extractor_prompt = read_text_file(self.ENTITY_EXTRACTOR_PROMPT_PATH)
        info(f"Loaded entity extractor prompt from {os.path.relpath(self.ENTITY_EXTRACTOR_PROMPT_PATH, PROJECT_ROOT)}.")

        all_num, success_num = 0, 0

        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
            futures = []
            for i, cur_input in enumerate(inputs):
                if "entity" not in cur_input:
                    futures.append(executor.submit(self.process_input, cur_input, entity_extractor_prompt, i))

            all_num = len(futures)

            iterator = tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True, desc="Extracting entities")

            for future in iterator:
                result, i = future.result(timeout=10*60)
                if result != None:
                    inputs[i] = result
                    success_num += 1

                    # Save intermediate results only if not in direct mode
                    if not direct_mode and success_num % self.SAVE_INTERVAL == 0:
                        save_json(inputs, self.ENTITY_EXTRACTOR_OUTPUT_PATH)
                        info(f"Saved checkpoint: {success_num}/{all_num} processed")

        # Only save if not in direct mode
        if not direct_mode:
            if success_num or not os.path.exists(self.ENTITY_EXTRACTOR_OUTPUT_PATH):
                info(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.ENTITY_EXTRACTOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                save_json(inputs, self.ENTITY_EXTRACTOR_OUTPUT_PATH)

        info(f"Total prompt tokens: {self.total_prompt_tokens}")
        info(f"Total completion tokens: {self.total_completion_tokens}")
        info(f"Total entities extracted: {self.total_entities}")
        info(f"Total relationships extracted: {self.total_relationships}")
        info(f"Total facts extracted: {self.total_entities + self.total_relationships}")
        if success_num > 0:
            info(f"Average entities per input: {self.total_entities/success_num:.2f}")
            info(f"Average relationships per input: {self.total_relationships/success_num:.2f}")
            info(f"Average facts per input: {(self.total_entities + self.total_relationships)/success_num:.2f}")
        info(f"Task Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Task Success rate: N/A")

        success_rate = success_num/all_num

        if direct_mode:
            # In direct mode, return the processed inputs instead of just stats
            return inputs, self.total_prompt_tokens, self.total_completion_tokens, success_rate, self.total_entities, self.total_relationships
        else:
            return self.total_prompt_tokens, self.total_completion_tokens, success_rate, self.total_entities, self.total_relationships

def extract_entity_from_output(input_text):
    """Regex String processing function to extract entities and relationships from model output"""
    entity_pattern = r'\("entity"\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\)'
    relationship_pattern = r'\("relationship"\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\)'

    # Replace placeholders with actual delimiters
    tuple_delimiter = '<tuple_delimiter>'
    record_delimiter = '<record_delimiter>'
    completion_delimiter = '<completion_delimiter>'

    # Parse entities
    entities = re.findall(entity_pattern, input_text)
    parsed_entities = [
        {
            "entity_name": match[0],
            "entity_description": match[1]
        }
        for match in entities
    ]

    # Parse relationships
    relationships = re.findall(relationship_pattern, input_text)
    parsed_relationships = []
    for match in relationships:
        # Extract sentence numbers from the sentences_used field
        sentences_raw = match[3]

        # Remove square brackets if present and split by comma
        sentences_clean = sentences_raw.strip("[]")

        numbers_and_ranges = re.findall(r'\d+-\d+|\d+', sentences_clean)

        # Create a formatted string for sentences_used
        formatted_sentence_numbers = ','.join(numbers_and_ranges)
        formatted_sentence_numbers = f'{formatted_sentence_numbers}'

        parsed_relationships.append({
            "source_entity_name": match[0].strip(),
            "target_entity_name": match[1].strip(),
            "relationship_description": match[2].strip(),
            "sentences_used": formatted_sentence_numbers
        })

    # Validate output format
    is_complete = completion_delimiter in input_text

    return {
        "entities": parsed_entities,
        "relationships": parsed_relationships,
        "is_complete": is_complete
    }

if __name__ == "__main__":
    entity_extractor = EntityExtractor()
    entity_extractor.run()
