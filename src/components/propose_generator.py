import os
import random
import re
from tqdm import tqdm
import threading
from src import PROJECT_ROOT
from src.components.entity_graph_constructor import EntityRelationshipGraph
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.api_utils import call_api_qwen
from src.utils.rag_utils import reformat_objective_facts, convert_set_to_list
from src.utils.file_utils import read_text_file, save_json, load_json
from src.utils.logger import info, error

class ProposeGenerator:
    def __init__(self):
        self.PROPOSE_GENERATOR_PROMPT_PATH = None
        self.PROPOSE_GENERATOR_GENERATED_TYPE = None
        self.PROPOSE_GENERATOR_INPUT_PATH = None
        self.PROPOSE_GENERATOR_OUTPUT_PATH = None
        self.PROPOSE_GENERATOR_GRAPH_PATH = None
        self.NUM_WORKERS = None
        self.SAVE_INTERVAL = None
        self.TEMPERATURE = None

        if os.getenv("PROPOSE_GENERATOR_CONTENT_INPUT_PATH", None) is not None:
            self.PROPOSE_GENERATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_CONTENT_INPUT_PATH"))
            self.PROPOSE_GENERATOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_CONTENT_PROMPT_PATH"))
            self.PROPOSE_GENERATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_CONTENT_OUTPUT_PATH"))
            self.PROPOSE_GENERATOR_SYSTEM_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, os.getenv("SYSTEM_PROMPT_PATH", None)))
            self.PROPOSE_GENERATOR_GENERATED_TYPE = "content"
        elif os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) is not None:
            self.PROPOSE_GENERATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH"))
            self.PROPOSE_GENERATOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_PROMPT_PATH"))
            self.PROPOSE_GENERATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_OUTPUT_PATH"))
            self.PROPOSE_GENERATOR_GRAPH_PATH = os.path.join(PROJECT_ROOT, os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_VISUALIZATION_PATH"))
            self.PROPOSE_GENERATOR_SYSTEM_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, os.getenv("SYSTEM_PROMPT_PATH", None)))
            self.PROPOSE_GENERATOR_GENERATED_TYPE = "entity_graph"
        else:
            raise EnvironmentError("Environment variables not configured properly for Propose Generator.")

        # Optional Parameters
        self.NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
        self.PROPOSE_GENERATOR_MAX_GEN_TIMES = int(os.getenv("MAX_QUESTION_GENERATED", 300))
        self.MAX_QUESTION = int(os.getenv("MAX_QUESTION_PER_CHUNK", 3))
        self.SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", 10))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

        # token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

        # question generation tracker
        self.total_questions_generated = 0

    def _entity_relationship_graph_2_entity_relationship_prompt(self, entity_relationship_graph, strategy="random_relationship"):
        
        already_have_chunks = set()
        objective_relationships = []
        objective_relationship_prompts = []
        for relationship_item in entity_relationship_graph['relationships']:
            chunk_id = relationship_item['id']
            if chunk_id in already_have_chunks:
                continue # Keeps only the first relationship from each unique chunk, discards the rest.
            already_have_chunks.add(chunk_id)
            objective_relationships.append(relationship_item)
            objective_relationship_prompts.append(f"<source_entity_name>{relationship_item['source_entity_name']}</source_entity_name>\n<target_entity_name>{relationship_item['target_entity_name']}</target_entity_name>\n<relationship_desc>{relationship_item['relationship_description']}</relationship_desc>")
        
        if len(objective_relationship_prompts) == 0:
            return "", [], []

        if strategy == "random_relationship":
            
            objective_idx_list = list(range(len(objective_relationship_prompts)))
            random_objective_idx_list = random.sample(objective_idx_list, min(10, len(objective_idx_list)))
            
            random_objective_relationships = [objective_relationships[idx] for idx in random_objective_idx_list]
            random_objective_relationship_prompts = [objective_relationship_prompts[idx] for idx in random_objective_idx_list]
            random_objective_relationship_prompts_with_numbers = [f"{idx+1}. {relationship_prompt}" for idx, relationship_prompt in enumerate(random_objective_relationship_prompts)]

            entity_relationship_prompt = "Objective Relationships:\n" + "\n".join(random_objective_relationship_prompts_with_numbers) + "\n\n"
            return entity_relationship_prompt, random_objective_relationship_prompts, random_objective_relationships
        else:
            raise NotImplementedError(f"Invalid value for 'strategy': {strategy}")

    def process_input_content(self, cur_input, cur_propose_generator_prompt, i):
        try:
            propose_generator_response, prompt_tokens, completion_tokens = call_api_qwen(cur_propose_generator_prompt, temperature=self.TEMPERATURE, system_prompt=self.PROPOSE_GENERATOR_SYSTEM_PROMPT)

            # Thread-safe token accumulation
            with self.token_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

            proposed_questions = extract_execution_output_content(propose_generator_response)

            # Count total questions generated
            questions_count = len(proposed_questions)

            result = {
                **cur_input, # python dictonary unpacking, 把所有的key展开
                'proposed-questions': proposed_questions
            }
            return result, i, questions_count
        except Exception as e:
            error(f"An error occurred while processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None, 0

    def run(self, inputs=None):

        info("=" * 100)
        info("Running Propose Generator".center(100))
        info("=" * 100)

        # Determine if we're in direct mode (inputs provided)
        direct_mode = inputs is not None

        # Propose generator prompt
        purpose_generator_prompt = read_text_file(self.PROPOSE_GENERATOR_PROMPT_PATH)
        # Replace max questions placeholder in the prompt
        if self.PROPOSE_GENERATOR_GENERATED_TYPE in ['content']:
            purpose_generator_prompt = purpose_generator_prompt.replace('[[MAX_QUESTIONS]]', str(self.MAX_QUESTION))
        elif self.PROPOSE_GENERATOR_GENERATED_TYPE in ['entity_graph']:
            purpose_generator_prompt = purpose_generator_prompt.replace('[[MAX_QUESTIONS]]', str(self.MAX_QUESTION))

        if not direct_mode:
            info(f"Loaded propose generator prompt from {os.path.relpath(self.PROPOSE_GENERATOR_PROMPT_PATH, PROJECT_ROOT)}.")

        if self.PROPOSE_GENERATOR_GENERATED_TYPE in ['content']:
            # Load inputs if not provided
            if inputs is None:
                if os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                    inputs = load_json(self.PROPOSE_GENERATOR_OUTPUT_PATH)
                    info(f"Loaded propose generator {len(inputs)} examples from {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.")
                else:
                    inputs = load_json(self.PROPOSE_GENERATOR_INPUT_PATH)
                    info(f"Loaded propose generator {len(inputs)} examples from {os.path.relpath(self.PROPOSE_GENERATOR_INPUT_PATH, PROJECT_ROOT)}.")
            else:
                pass  # Direct mode - inputs provided

            success_num, all_num = 0, 0
            tasks = []

            self.executor = ThreadPoolExecutor(max_workers=self.NUM_WORKERS)
            try:
                for i, cur_input in enumerate(inputs[:self.PROPOSE_GENERATOR_MAX_GEN_TIMES]):
                    if 'proposed-questions' in cur_input:
                        continue
                    # Reformat facts would look like: <detailed-desc>Many hostels have strict smoking policies that prohibit indoor smoking to maintain a safe and healthy environment.</detailed-desc>
                    context = reformat_objective_facts(cur_input)
                    cur_propose_generator_prompt = purpose_generator_prompt.replace('[[CONTEXT]]', context)
                    future = self.executor.submit(self.process_input_content, cur_input, cur_propose_generator_prompt, i)
                    tasks.append(future)

                all_num = len(tasks)
                for future in tqdm(as_completed(tasks), total=len(tasks), desc="Generating Questions...", dynamic_ncols=True):
                    try:
                        result, i, questions_count = future.result(timeout=10*60)

                        inputs[i] = result

                        # Track total questions generated with thread safety
                        with self.token_lock:
                            self.total_questions_generated += questions_count

                        success_num += 1
                        if not direct_mode and success_num % self.SAVE_INTERVAL == 0:
                            info(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                            save_json(inputs, self.PROPOSE_GENERATOR_OUTPUT_PATH)
                    except Exception as e:
                        error(f"Error during processing: {e}")
            except KeyboardInterrupt:
                info("Task interrupted by user")
            finally:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None

            # Only save if not in direct mode
            if not direct_mode:
                if success_num or not os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                    info(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                    save_json(inputs, self.PROPOSE_GENERATOR_OUTPUT_PATH)

            info(f"Total prompt tokens: {self.total_prompt_tokens}")
            info(f"Total completion tokens: {self.total_completion_tokens}")
            info(f"Total questions generated: {self.total_questions_generated}")
            info(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

            if direct_mode:
                return inputs, self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num, self.total_questions_generated
            else:
                return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num, self.total_questions_generated

        elif self.PROPOSE_GENERATOR_GENERATED_TYPE in ['entity_graph']:
            # Load inputs if not provided
            if inputs is None:
                inputs = load_json(self.PROPOSE_GENERATOR_INPUT_PATH)
                info(f"Loaded propose generator {len(inputs)} examples from {os.path.relpath(self.PROPOSE_GENERATOR_INPUT_PATH, PROJECT_ROOT)}.")
            else:
                pass  # Direct mode - inputs provided

            # Build entity relationship graph
            entity_relationship_graph = EntityRelationshipGraph(inputs)

            # Save graph visualization if not in direct mode
            if not direct_mode:
                entities_dict = {}
                for entity_id, entity_info in entity_relationship_graph.entities.items():
                    entities_dict[entity_id] = {
                        **entity_info,
                        'chunk_names': list(entity_info['chunk_names']) if isinstance(entity_info['chunk_names'], set) else entity_info['chunk_names']
                    }
                save_json(entities_dict, self.PROPOSE_GENERATOR_GRAPH_PATH)
                info(f"Saved entity relationship graph to {os.path.relpath(self.PROPOSE_GENERATOR_GRAPH_PATH, PROJECT_ROOT)}.")

            outputs = {}
            if not direct_mode and os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                outputs = load_json(self.PROPOSE_GENERATOR_OUTPUT_PATH)
                info(f"Loaded {len(outputs)} outputs from {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.")

            already_done_entity_ids = set(outputs.keys())
            already_done_entity_ids = set([int(entity_id) for entity_id in already_done_entity_ids])

            all_num, success_num = 0, 0
            tasks = []

            self.executor = ThreadPoolExecutor(max_workers=self.NUM_WORKERS)
            try:
                for cur_entity_id, cur_entity_item in list(entity_relationship_graph.items())[:self.PROPOSE_GENERATOR_MAX_GEN_TIMES]:
                    if cur_entity_id in already_done_entity_ids:
                        continue
                    public_entity_name = cur_entity_item['entity_name']

                    subgraph_depth_1 = entity_relationship_graph.get_subgraph(cur_entity_id, depth=1)
                    subgraph_depth_1 = convert_set_to_list(subgraph_depth_1)
                    entity_relationship_prompt, \
                        cur_objective_relationship_prompts, \
                            cur_objective_relationships = self._entity_relationship_graph_2_entity_relationship_prompt(subgraph_depth_1)

                    if entity_relationship_prompt == "" or len(cur_objective_relationship_prompts) <= 1:
                        continue

                    cur_propose_generator_prompt = purpose_generator_prompt.replace('[[ENTITY_NAME]]', public_entity_name)
                    cur_propose_generator_prompt = cur_propose_generator_prompt.replace('[[CONTEXT]]', entity_relationship_prompt)
                    future = self.executor.submit(call_api_qwen, cur_propose_generator_prompt, temperature=self.TEMPERATURE, system_prompt=self.PROPOSE_GENERATOR_SYSTEM_PROMPT)
                    tasks.append((future, cur_entity_id, subgraph_depth_1, cur_objective_relationships, cur_objective_relationship_prompts))

                all_num = len(tasks)
                for future in tqdm(as_completed([t[0] for t in tasks]), total=len(tasks), desc="Generating Questions...", dynamic_ncols=True):
                    idx = [t[0] for t in tasks].index(future)
                    if idx == -1:
                        raise ValueError("Invalid index.")
                    cur_entity_id, subgraph_depth_1, cur_objective_relationships, cur_objective_relationship_prompts = tasks[idx][1], tasks[idx][2], tasks[idx][3], tasks[idx][4]

                    try:
                        propose_generator_response, prompt_tokens, completion_tokens = future.result(timeout=10*60)

                        # Thread-safe token accumulation
                        with self.token_lock:
                            self.total_prompt_tokens += prompt_tokens
                            self.total_completion_tokens += completion_tokens

                        tmp_proposed_questions = extract_execution_output_entity_graph(propose_generator_response)
                        proposed_questions = {}
                        for tmp_proposed_question_type, tmp_proposed_question_dict in tmp_proposed_questions.items():
                            if "objective-relationship-id" in tmp_proposed_question_dict:
                                objective_relationship_ids = re.findall(r'\d+-\d+|\d+', tmp_proposed_question_dict["objective-relationship-id"])
                                actual_number_count = count_actual_numbers(objective_relationship_ids)
                                if actual_number_count > 1:
                                    proposed_questions[tmp_proposed_question_type] = tmp_proposed_question_dict
                                else:
                                    continue
                            else:
                                continue

                        outputs[cur_entity_id] = ({
                            'relationships': subgraph_depth_1['relationships'],
                            'selected-relationships': {
                                'objective-relationships': cur_objective_relationships,
                                'objective-relationship-prompts': cur_objective_relationship_prompts,
                            },
                            'proposed-questions': proposed_questions
                        })

                        # Track total questions generated with thread safety
                        with self.token_lock:
                            self.total_questions_generated += len(proposed_questions)

                        success_num += 1
                        if not direct_mode and success_num % self.SAVE_INTERVAL == 0:
                            save_json(outputs, self.PROPOSE_GENERATOR_OUTPUT_PATH)
                    except Exception as e:
                        error(f"Error processing entity {cur_entity_id}: {e}")
            except KeyboardInterrupt:
                info("Task interrupted by user")
            finally:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None

            # Only save if not in direct mode
            if not direct_mode:
                if success_num or not os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                    info(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                    save_json(outputs, self.PROPOSE_GENERATOR_OUTPUT_PATH)

            info(f"Total prompt tokens: {self.total_prompt_tokens}")
            info(f"Total completion tokens: {self.total_completion_tokens}")
            info(f"Total questions generated: {self.total_questions_generated}")
            info(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

            if direct_mode:
                return outputs, self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num, self.total_questions_generated
            else:
                return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num, self.total_questions_generated

def extract_execution_output_content(text):
    """
    Extracts the structured content of "Execution" output dynamically for any question categories.

    Args:
        text (str): Input text containing "Execution" output.

    Returns:
        dict: A dictionary containing structured data for each dynamically matched question category.
    """
    # Define regex to capture all question categories and their content
    category_pattern = re.compile(r"\d+\.\s*<([^>]+-questions)>(.*?)((?=\d+\.\s*<[^>]+-questions>)|$)", re.S)

    # Parse content for each category dynamically
    categories = {}
    for match in category_pattern.finditer(text):
        category_name = match.group(1).strip()
        category_content = match.group(2).strip()

        # Function to parse individual questions within a category
        def parse_questions(content):
            question_pattern = re.compile(
                r"<question>(.*?)</question>\s*<objective-facts>(.*?)</objective-facts>\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>",
                re.S
            )
            ret = [
                {
                    "question": match.group(1).strip(),
                    "objective-facts": match.group(2).strip(),
                    "reasoning": match.group(3).strip(),
                    "answer": match.group(4).strip()
                }
                for match in question_pattern.finditer(content)
            ]
            if len(ret) == 0:
                return []
            return ret

        # Parse and store questions for the current category
        ret = parse_questions(category_content)
        if len(ret) == 0:
            continue
        categories[category_name] = ret[0]

    return categories

def extract_execution_output_entity_graph(text):
    """
    Extracts the structured content of "Execution" output dynamically for any question categories.

    Args:
        text (str): Input text containing "Execution" output.

    Returns:
        dict: A dictionary containing structured data for each dynamically matched question category.
    """
    # Define regex to capture all question categories and their content
    category_pattern = re.compile(r"\d+\.\s*<([^>]+-questions)>(.*?)((?=\d+\.\s*<[^>]+-questions>)|$)", re.S)

    # Parse content for each category dynamically
    categories = {}
    for match in category_pattern.finditer(text):
        category_name = match.group(1).strip()
        category_content = match.group(2).strip()

        # Function to parse individual questions within a category
        def parse_questions(content):
            question_pattern = re.compile(
                r"<question>(.*?)</question>\s*<objective-relationship-id>(.*?)</objective-relationship-id>\s*"
                r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>",
                re.S
            )
            ret = [
                {
                    "question": match.group(1).strip(),
                    "objective-relationship-id": match.group(2).strip(),
                    "reasoning": match.group(3).strip(),
                    "answer": match.group(4).strip()
                }
                for match in question_pattern.finditer(content)
            ]
            if len(ret) == 0:
                return []
            return ret

        # Parse and store questions for the current category
        ret = parse_questions(category_content)
        if len(ret) == 0:
            continue
        categories[category_name] = ret[0]

    return categories

def count_actual_numbers(numbers_and_ranges):
    total_count = 0
    for item in numbers_and_ranges:
        if '-' in item:  # It's a range like 'xx1-xx2'
            start, end = map(int, item.split('-'))
            total_count += (end - start + 1)
        else:  # It's a single number
            total_count += 1
    return total_count