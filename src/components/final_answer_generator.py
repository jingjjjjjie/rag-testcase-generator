import re
import os
import json
import signal
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src import PROJECT_ROOT
from src.utils.file_utils import load_json,save_json,read_text_file
from src.utils.api_utils import call_api_qwen
from src.utils.logger import info, error
from src.utils.rag_utils import (
    reformat_objective_facts,
    extract_largest_json,
    convert_set_to_list,
    expand_numbers_and_ranges,
    replace_clue_with_doc_and_sen,
    list_to_docided_string)

class FinalAnswerGenerator:
    def __init__(self):

        info("=" * 100)
        info("Running Final Answer Generator".center(100))
        info("=" * 100)

        self.FINAL_ANSWER_GENERATOR_INPUT_PATH, self.FINAL_ANSWER_GENERATOR_CORPUS_PATH, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = None, None, None, None

        if os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH", None) != None:
            self.FINAL_ANSWER_GENERATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH"))
            self.FINAL_ANSWER_GENERATOR_CORPUS_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_CORPUS_PATH"))
            self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_OUTPUT_PATH"))
            self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_PROMPT_PATH"))
            self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE = 'content'
        elif os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.FINAL_ANSWER_GENERATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH"))
            self.FINAL_ANSWER_GENERATOR_CORPUS_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_CORPUS_PATH"))
            self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_OUTPUT_PATH"))
            self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, os.getenv("FINAL_ANSWER_GENERATOR_PROMPT_PATH"))
            self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE = 'entity_graph'
        else:
            raise EnvironmentError("Environment variables is not defined properly.")

        # Optional parameters
        self.NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
        self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES = int(os.getenv("FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES", 300))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))
        self.SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", 10))

        # Token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

        # Executor for signal handling
        self.executor = None
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Load prompt template (mandatory)
        self.prompt_template = read_text_file(self.FINAL_ANSWER_GENERATOR_PROMPT_PATH)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C and termination signals"""
        info("Received shutdown signal, cancelling tasks...")
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        raise KeyboardInterrupt()

    def process_input_content(self, cur_input, cur_prompt):
        try:
            cur_response, prompt_tokens, completion_tokens = call_api_qwen(cur_prompt, temperature=self.TEMPERATURE)

            # Thread-safe token accumulation
            with self.token_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

            answers = extract_answers(cur_response)
            cur_input['positive'] = answers['short-answer']['answer']
            cur_input['corrected-answer'] = answers
            return cur_input
        except Exception as e:
            error(f"An error occurred while processing input: {e}")
            return None

    def run(self, inputs=None, corpus_data=None):
        # Determine if we're in direct mode (inputs provided)
        direct_mode = inputs is not None

        # Load inputs if not provided
        if not direct_mode:
            if os.path.exists(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH):
                inputs = load_json(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)
                info(f"Loaded final answer generator {len(inputs)} examples from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.")
            else:
                inputs = load_json(self.FINAL_ANSWER_GENERATOR_INPUT_PATH)
                info(f"Loaded final answer generator {len(inputs)} examples from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_INPUT_PATH, PROJECT_ROOT)}.")

        # Load prompt template
        prompt_template = read_text_file(self.FINAL_ANSWER_GENERATOR_PROMPT_PATH)
        if not direct_mode:
            info(f"Loaded prompt template from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_PROMPT_PATH, PROJECT_ROOT)}.")

        # Load corpus, and build context dictionary
        if corpus_data is None:
            corpus_data = load_json(self.FINAL_ANSWER_GENERATOR_CORPUS_PATH)
            if not direct_mode:
                info(f"Loaded corpus with {len(corpus_data)} examples from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_CORPUS_PATH, PROJECT_ROOT)}.")
        corpusid_2_context = {cur_dict['id']: cur_dict['context'] for cur_dict in corpus_data}


        success_num, all_num = 0, 0
        futures_to_data = {}
        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
            if self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE in ['content']:
                # process content chunk by chunk
                data_list = inputs[:self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES]
                for data_item in data_list:
                        # Skip the current chunk if proposed-questions is not present (propose_generator not run properly)
                        if 'proposed-questions' not in data_item:
                            continue
                        proposed_questions = data_item['proposed-questions']
                        chunk_id = data_item['id']  # for example，in our case: "doc1_question1"

                        all_clueid2docid2senidlist = {}
                        objective_facts = data_item['objective-facts']
                        sens = data_item["sens"]
                        for (fact_id, objective_fact), sen in zip(enumerate(objective_facts, start=1), sens):
                            sen_ids = re.findall(r'\d+-\d+|\d+', sen) # extract csentences such as "5-8" or "10,11"
                            sen_ids = expand_numbers_and_ranges(sen_ids) # "5-8" → [5, 6, 7, 8]
                            all_clueid2docid2senidlist[fact_id] = {
                                chunk_id: sen_ids
                            }

                        for proposed_question_type, proposed_question_dict in proposed_questions.items():
                            if "positive" in proposed_question_dict:
                                continue
                            # get answer with already replaced clues
                            original_question = proposed_question_dict['question']
                            positive_answer = proposed_question_dict['answer']
                            if not positive_answer:
                                continue

                            # print("all_clueid2docid2senidlist:", all_clueid2docid2senidlist)
                            # print("positive_answer:", positive_answer)
                            positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer)

                            needed_corpusid2corpus = {chunk_id: corpusid_2_context[chunk_id]}
                            needed_corpusid2corpus_str = list_to_docided_string(needed_corpusid2corpus)

                            cur_prompt = self.prompt_template.replace('[[QUESTION]]', original_question)
                            cur_prompt = cur_prompt.replace('[[CONTEXT]]', needed_corpusid2corpus_str)
                            cur_prompt = cur_prompt.replace('[[ANSWER]]', positive_answer)
                            future = executor.submit(self.process_input_content, proposed_question_dict, cur_prompt)
                            futures_to_data[future] = (
                                None
                            )

            elif self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE in ['entity_graph']:
                data_list = list(inputs.values())[:self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES]
                for data_item in data_list:
                    if 'proposed-questions' not in data_item:
                        continue
                    proposed_questions = data_item['proposed-questions']
                    objective_relationships = data_item['selected-relationships']['objective-relationships']
                    objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

                    all_clueid2docid2senidlist = {}
                    for (relationship_id, objective_relationship_dict) in enumerate(objective_relationships, start=1):
                        docid = objective_relationship_dict['id']
                        sen_ids = re.findall(r'\d+-\d+|\d+', objective_relationship_dict["sentences_used"])
                        sen_ids = expand_numbers_and_ranges(sen_ids)
                        all_clueid2docid2senidlist[relationship_id] = {
                            docid: sen_ids
                        }
                    for proposed_question_type, proposed_question_dict in proposed_questions.items():
                        if "positive" in proposed_question_dict:
                            continue
                        # get answer with already replaced clues
                        original_question = proposed_question_dict['question']
                        positive_answer = proposed_question_dict['answer']
                        # print("all_clueid2docid2senidlist:", all_clueid2docid2senidlist)
                        # print("positive_answer:", positive_answer)
                        positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer)

                        # get needed chunk ids
                        needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                        needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                        needed_related_relationships = [objective_relationship_id_2_objective_relationship[int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship]
                        needed_corpusids = []
                        for relationship_id in needed_objective_relationship_ids:
                            if relationship_id in all_clueid2docid2senidlist:
                                needed_corpusids.extend(list(all_clueid2docid2senidlist[relationship_id].keys()))
                        needed_corpusids = list(sorted(list(set(needed_corpusids))))
                        needed_corpusid2corpus = {
                            cur_related_relationship['id']: corpusid_2_context[cur_related_relationship['id']]
                            for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship and cur_related_relationship['id'] in corpusid_2_context
                        }

                        needed_corpusid2corpus_str = list_to_docided_string(needed_corpusid2corpus)
                        cur_prompt = self.prompt_template.replace('[[QUESTION]]', original_question)
                        cur_prompt = cur_prompt.replace('[[CONTEXT]]', needed_corpusid2corpus_str)
                        cur_prompt = cur_prompt.replace('[[ANSWER]]', positive_answer)
                        future = executor.submit(self.process_input_content, proposed_question_dict, cur_prompt)
                        futures_to_data[future] = (
                            None
                        )
            else:
                raise ValueError(f"Unknown data file: {self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE}")

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=all_num, desc="Evaluating Test-Cases", dynamic_ncols=True):
                # rephrased_questions, rephrased_questions_part, rephrased_questions_hybrid = futures_to_data[future]
                _ = futures_to_data[future]
                try:
                    cur_response = future.result(timeout=10*60)

                    if cur_response is not None:
                        success_num += 1
                        if not direct_mode and (success_num + 1) % self.SAVE_INTERVAL == 0:
                            info(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                            save_json(inputs, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)
                except Exception as e:
                    error(f"Error processing future: {e}")

        # Only save if not in direct mode
        if not direct_mode:
            if success_num or not os.path.exists(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH):
                info(f'Saving outputs to {os.path.relpath(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                save_json(inputs, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)

        info(f"Total prompt tokens: {self.total_prompt_tokens}")
        info(f"Total completion tokens: {self.total_completion_tokens}")
        info(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

        if direct_mode:
            # In direct mode, return the processed inputs along with stats
            return inputs, self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num
        else:
            return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num

def extract_answers(prompt):
    # Define regular expressions to capture the short and long answers
    short_answer_pattern = r"<answer-short>\s*<reason>(.*?)</reason>\s*<answer>(.*?)</answer>\s*</answer-short>"
    #long_answer_pattern = r"<answer-long>\s*<reason>(.*?)</reason>\s*<answer>(.*?)</answer>\s*</answer-long>"

    # Search for the patterns in the prompt
    short_answer_match = re.search(short_answer_pattern, prompt, re.DOTALL)
    #long_answer_match = re.search(long_answer_pattern, prompt, re.DOTALL)

    # Extract the reason and answer for short and long answers
    if short_answer_match:
        short_reason = short_answer_match.group(1).strip()
        short_answer = short_answer_match.group(2).strip()
    else:
        short_reason = None
        short_answer = None

    # if long_answer_match:
    #     long_reason = long_answer_match.group(1).strip()
    #     long_answer = long_answer_match.group(2).strip()
    # else:
    #     long_reason = None
    #     long_answer = None

    return {
        "short-answer": {
            "reason": short_reason,
            "answer": short_answer
        }
        # "long-answer": {
        #     "reason": long_reason,
        #     "answer": long_answer
        # }
    }
