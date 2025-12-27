from typing import Dict, List
import json
import re

def reformat_objective_facts(data):
    """
    Reformat objective facts from a dictionary into a structured string format.

    Takes a dictionary containing 'objective-facts' (a list of fact strings)
    and formats them into a numbered, XML-tagged string suitable for prompts
    or downstream processing.

    Args:
        data (dict): Dictionary with key 'objective-facts' containing a list of fact strings
                    Example: {"objective-facts": ["fact1", "fact2", "fact3"]}

    Returns:
        str: Formatted string with numbered facts wrapped in XML tags

    Example:
        Input: {"objective-facts": ["The sky is blue", "Water is wet"]}
        Output:
            "Objective Facts:
            1. <detailed-desc>The sky is blue</detailed-desc>
            2. <detailed-desc>Water is wet</detailed-desc>\n"
    """
    # Initialize result dictionary to store formatted facts
    result = {"Objective Facts": []}

    # Reformat each objective fact with numbering and XML tags
    for idx, fact in enumerate(data['objective-facts'], start=1):
        # Format: "1. <detailed-desc>fact text</detailed-desc>"
        result["Objective Facts"].append(
            f"{idx}. <detailed-desc>{fact}</detailed-desc>"
        )

    # Convert the dictionary to a formatted string
    result_str = ""
    for key, values in result.items():
        # Add the header "Objective Facts:"
        result_str += f"{key}:\n"
        # Join all facts with newlines and add a final newline
        result_str += "\n".join(values) + "\n"

    return result_str

def list_to_numbered_string(string_list):
    """
    Convert a list of strings into a numbered string.

    :param string_list: list of str, the list of strings to be converted
    :return: str, the resulting numbered string
    """
    numbered_string = ""
    for index, string in enumerate(string_list, start=1):
        numbered_string += f"{index}. {string}\n"
    return numbered_string.strip()

def convert_set_to_list(obj):
    """
    Recursively convert sets to lists in a nested structure.

    This is necessary because JSON format doesn't support Python sets,
    so we need to convert them to lists before serialization.

    Args:
        obj: Any Python object (set, dict, list, or primitive type)

    Returns:
        The same object structure with all sets converted to lists

    Example:
        Input: {"items": {1, 2, 3}, "nested": {"vals": {4, 5}}}
        Output: {"items": [1, 2, 3], "nested": {"vals": [4, 5]}}
    """
    # Base case: if object is a set, convert to list
    if isinstance(obj, set):
        return list(obj)

    # Recursive case: if dictionary, process all key-value pairs
    elif isinstance(obj, dict):
        return {key: convert_set_to_list(value) for key, value in obj.items()}

    # Recursive case: if list, process all elements
    elif isinstance(obj, list):
        return [convert_set_to_list(item) for item in obj]

    # Base case: for primitives (int, str, bool, etc.), return as-is
    else:
        return obj

def expand_numbers_and_ranges(numbers_and_ranges):
    expanded_numbers = []
    for item in numbers_and_ranges:
        if '-' in item:  # It's a range like 'xx1-xx2'
            start, end = map(int, item.split('-'))
            if start > end:
                end, start = start, end
            expanded_numbers.extend(range(start, end + 1))
        else:  # It's a single number
            expanded_numbers.append(int(item))
    expanded_numbers = list(sorted(list(set(expanded_numbers))))
    return expanded_numbers

def replace_clue_with_doc_and_sen(all_clueid2docid2senidlist: Dict[int, Dict[int, List[int]]], positive_answer: str) -> str:
    """
    Replaces [Clue xx] or [Clue xx-yy] citations in the positive_answer with formatted [Doc xx, Sen xx] citations.
    
    Parameters:
    - all_clueid2docid2senidlist: Dict mapping clue IDs to another dict mapping doc IDs to lists of sentence IDs.
      Example:
      {
          1: {1: [1, 2, 3]},
          2: {1: [4, 5]},
          3: {2: [1, 2]},
          4: {2: [3]},
      }
    - positive_answer: String containing [Clue xx] or [Clue xx-yy] patterns.
    
    Returns:
    - new_answer: String with [Clue xx] patterns replaced by formatted citations.
    """
    
    def expand_range(token: str) -> List[int]:
        """
        Expands a token which can be a single number or a range (e.g., '2' or '2-8') into a list of integers.
        """
        if '-' in token:
            start, end = token.split('-')
            return list(range(int(start), int(end) + 1))
        else:
            return [int(token)]
    
    def expand_range_in_list(tokens: List[str]) -> List[int]:
        """
        Processes a list of tokens, expanding ranges and collecting all clue IDs.
        """
        clue_ids = []
        for token in tokens:
            if '-' in token:
                clue_ids.extend(expand_range(token))
            else:
                if token.isdigit():
                    clue_ids.append(int(token))
        return clue_ids
    
    def expand_sen_ranges(nums: List[int]) -> List[str]:
        """
        Converts a sorted list of integers into a list with ranges for consecutive numbers.
        Example: [1,2,3,5] -> ['1-3', '5']
        """
        if not nums:
            return []

        nums = sorted(nums)
        ranges = []
        start = prev = nums[0]

        for num in nums[1:]:
            if num == prev + 1:
                prev = num
            else:
                if prev - start >= 2:
                    ranges.append(f"{start}-{prev}")
                elif prev - start == 1:
                    ranges.append(str(start))
                    ranges.append(str(prev))
                else:
                    ranges.append(str(start))
                start = prev = num

        # Handle the last range
        if prev - start >= 2:
            ranges.append(f"{start}-{prev}")
        elif prev - start == 1:
            ranges.append(str(start))
            ranges.append(str(prev))
        else:
            ranges.append(str(start))

        return ranges

    # Regular expression to find [Clue xx], [Clue xx, yy], [Clue xx-yy], etc.
    clue_pattern = re.compile(r'\[Clue\s+([^\]]+)\]')
    
    def replacement(match):
        # print("match:", match)
        clue_ids_str = match.group(1)
        # Split by comma and/or whitespace
        tokens = re.split(r'[,\s]+', clue_ids_str)
        # Expand tokens to individual clue IDs
        clue_ids = expand_range_in_list(tokens)
        # print("clue_ids:", clue_ids)
        
        # Map doc_id to set of sen_ids
        doc_to_sens = {}
        for cid in clue_ids:
            if cid in all_clueid2docid2senidlist:
                for doc_id, sen_ids in all_clueid2docid2senidlist[cid].items():
                    if doc_id not in doc_to_sens:
                        doc_to_sens[doc_id] = set()
                    doc_to_sens[doc_id].update(sen_ids)
        
        if not doc_to_sens:
            # No valid clues found, return the original string
            return match.group(0)
        
        # Build the citation strings
        citations = []
        for doc_id in sorted(doc_to_sens.keys()):
            sen_list = sorted(doc_to_sens[doc_id])
            sen_ranges = expand_sen_ranges(sen_list)

            if sen_ranges:
                # Prepend 'Sen ' to each range
                sen_formatted = [f"{s}" for s in sen_ranges]
                sen_formatted[0] = f"Sen {sen_formatted[0]}"
                # Join sentence parts with comma
                sen_str = ", ".join(sen_formatted)
                citations.append(f"Doc {doc_id}, {sen_str}")
            else:
                citations.append(f"")
        
        # Format multiple documents with separate brackets
        if len(citations) == 1:
            return f"[{citations[0]}]"
        else:
            # Each document citation in its own brackets
            return "".join(f"[{cit}]" for cit in citations)
    
    # Replace all [Clue ...] patterns using the replacement function
    new_answer = clue_pattern.sub(replacement, positive_answer)
    
    return new_answer

def list_to_docided_string(string_dict):
    """
    Convert a list of strings into a docided string.

    :param string_list: list of str, the list of strings to be converted
    :return: str, the resulting numbered string
    """
    numbered_string = ""
    for index, (doc_id, doc_content) in enumerate(string_dict.items()):
        numbered_string += f"""{index}. <doc>
    <doc-name>{doc_id}</doc-name>
    <detailed-desc>{doc_content}</detailed-desc>
</doc>
"""
    return numbered_string.strip()

def extract_largest_json(response):
    """
    Extract the largest valid JSON object from a string response using a stack-based approach.

    LLMs often return JSON wrapped in markdown code blocks or with explanatory text.
    This function finds all valid JSON objects/arrays in the response and returns
    the largest one (most likely to be the main response).

    Algorithm:
    1. Use a stack to track matching brackets: { } [ ]
    2. Whenever brackets close correctly, extract that substring as potential JSON
    3. Try parsing each potential JSON and keep the largest valid one

    Args:
        response (str): The string response from the language model, which may contain
                       JSON mixed with other text or markdown formatting.

    Returns:
        dict or list: The extracted and parsed JSON object or array.

    Raises:
        ValueError: If no valid JSON object is found in the response.

    Example:
        Input: "Here's the data: ```json\n{\"key\": \"value\"}\n```"
        Output: {"key": "value"}
    """
    # Stack to track opening brackets and their positions
    stack = []  # Stores the bracket characters: '{' or '['
    start_indices = []  # Stores the index positions of opening brackets
    potential_jsons = []  # Collects all complete bracket-matched substrings

    # Iterate through each character in the response
    for i, char in enumerate(response):
        # When we find an opening bracket, push it onto the stack
        if char == '{' or char == '[':
            stack.append(char)  # Remember which type of bracket
            start_indices.append(i)  # Remember where it started

        # When we find a closing bracket, try to match with opening bracket
        elif char == '}' or char == ']':
            if stack:  # Only process if we have opening brackets waiting
                # Pop the most recent opening bracket from stack
                opening_bracket = stack.pop()
                start_index = start_indices.pop()

                # Check if brackets match correctly: {} or []
                if (opening_bracket == '{' and char == '}') or (opening_bracket == '[' and char == ']'):
                    # Extract the complete substring from opening to closing bracket
                    potential_json = response[start_index:i+1]
                    potential_jsons.append(potential_json)
            else:
                # Unmatched closing bracket (no opening bracket before it), skip it
                continue

    # Now parse all potential JSON strings and find the largest valid one
    largest_json = None  # Will store the parsed JSON object
    largest_size = 0  # Track the size in characters

    for potential_json in potential_jsons:
        try:
            # Try to parse the string as JSON
            parsed_json = json.loads(potential_json)
            size = len(potential_json)  # Get the length of the JSON string

            # If this is larger than previous ones, keep it
            if size > largest_size:
                largest_json = parsed_json
                largest_size = size

        except json.JSONDecodeError:
            # This substring wasn't valid JSON after all, skip it
            continue

    # If we didn't find any valid JSON, raise an error
    if largest_json == None:
        raise ValueError("No valid JSON object found in the response.")

    return largest_json