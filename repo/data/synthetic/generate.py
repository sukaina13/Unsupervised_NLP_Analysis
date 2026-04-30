# ========================
# Environment Configuration
# ========================
import os
import sys
from dotenv import load_dotenv
load_dotenv()

repo_root = os.getcwd()
sys.path.insert(0, os.path.join(repo_root, 'src'))

from groq import Groq
import pandas as pd
import numpy as np
import json
import random
import re
from tqdm import tqdm
import os

key = os.getenv('GROQ_API_KEY')
client = Groq()
model = 'openai/gpt-oss-120b'

def generate_hierarchy(topics, theme, terms=None, depth=2, temperature=0.7, model=model, num=3,with_synonyms=0,branching="constant",max_num=20):
    """
    Generate hierarchical data using the Groq API while maintaining context across API calls.
    """
    hierarchy = {}
    seen_nodes = set()
    
    messages_init = [
        {"role": "system", "content": """
            You are an expert in Social-Ecological Systems (SES) and systems dynamics. You generate hierarchies of measurable variables (stocks and flows), NOT abstract themes.
                Constraint 1 (Measurability): Every "subtopic" must be a variable that can increase or decrease (e.g., use "Nitrogen Runoff Rate" instead of "Pollution"; use "Household Trust Score" instead of "Social Capital").
                Constraint 2 (Distinction): Variables must be distinct. "Fish Biomass" and "Fish Count" are too similar; choose one.
                Task: Expand the Root Variable: "{ROOT_VARIABLE}" (e.g., Coastal Resilience) into a 3-level hierarchy.
                Level 1: Broad measurable dimensions (e.g., Economic Resource Availability, Ecological Diversity Index).
                Level 2: Specific component variables (e.g., Local Employment Rate, Mangrove Area Coverage).
                Level 3: Precise metric indicators (e.g., Percentage of Households with Savings, Sapling Density per Hectare).
                Output Format: JSON Tree
        """}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages_init,
        temperature=temperature
    )
    max_num=max_num

    content_init = response.choices[0].message.content.strip()
    messages_init.append({"role": "assistant", "content": content_init})

    def get_dynamic_num(level):
        """ Adjust number of children based on branching strategy with a smooth decrease. """
        if branching == "constant":
            return num
        elif branching == "decreasing":
            # Scale `num` down based on the proportion of depth remaining, ensuring a minimum of 2
            return max(2, round(num * (1 - (level - 1) / depth)))
        elif branching == "increasing":
            nonlocal max_num
            if max_num is None:
                max_num = num * 2  # Default to twice the starting value if no max is provided
            return min(max_num, max(2, round(num + (max_num - num) * (level - 1) / (depth - 1)))) 
        elif branching == "random":
            return random.randint(2, max(2, 2 * num - 2))

    def parse_json_response(response_text):
        """ Extract and parse JSON safely from model output """
        try:
            json_data = json.loads(response_text)
            if isinstance(json_data, list):
                return json_data
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            try:
                json_data = json.loads(match.group(0))
                if isinstance(json_data, list):
                    return json_data
            except json.JSONDecodeError:
                pass

        extracted = re.findall(r'^\d+\.\s*(.+)$', response_text, re.MULTILINE) 
        if not extracted:
            extracted = re.findall(r'^\*\s*(.+)$', response_text, re.MULTILINE)

        if extracted:
            return extracted

        print(f"Warning: Failed to parse valid JSON from response: {response_text}")
        return []

    def get_synonyms(topic, model=model, num=3):
        """Generate synonyms for a given topic using Groq and store separately."""
        messages = [
            {"role": "system", "content": "You provide synonyms and related phrases for a given term."},
            {"role": "user", "content": f"Generate {num} synonyms or closely related terms for '{topic}', in a JSON list format."}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        
        response_content = response.choices[0].message.content.strip()

        try:
            # Attempt to parse the response as a JSON array
            return json.loads(response_content)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract the list manually
            list_pattern = r'(\[.*\])'
            match = re.search(list_pattern, response_content)
            
            if match:
                try:
                    # Attempt to load the matched part as JSON
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
                
            # As a last resort, manually extract comma-separated items (in case no valid JSON is provided)
            # Fix the fallback pattern to avoid capturing quotes
            fallback_pattern = r'[“"]([^”"]+)[”"]'  # Match text inside either “ ” or " "
            fallback_matches = re.findall(fallback_pattern, response_content)

            # Clean the fallback matches and return them as a list
            if fallback_matches:
                return [item.strip() for item in fallback_matches[:num]]
            
            # If everything fails, return an empty list
            print("Warning: Unable to extract synonyms.")
            return []

    def expand_topic(topic, parent_dict, level, sibling_topics, core_topic, other_core_topics):
        """ Recursively expand a topic while maintaining API call context (without expanding synonyms). """
        if level > depth:
            return
        nonlocal messages_init
        nonlocal content_init

        current_num = get_dynamic_num(level)
        print(f"\n{'  ' * (level - 1)}[Level {level}] Expanding: {topic}")
        
        # Generate subtopics first (without including synonyms yet)
        prompt = f"""
            Expand the given **Parent Topic** into a list of up to **{current_num}** or less specific **subtopics** that:  
            - Are related to **{core_topic}**, especially **{topic}**.
            - Clearly reflect their connection to the **Parent Topic** and **Core Topic** through their nature and scope. 
            - Are **Not** related to the other core topics: **{other_core_topics}**.  
            - Much Less related to sibling topics: **{sibling_topics}**.  
            - Fit within the given **theme**: **{theme}**. 
            - Each subtopic should be a **subcategory** of the parent topic, such that a change in the subtopic implies a change in the broader topic.  

            **Parent Topic:** {topic} 
            **Core Topic:** {core_topic}

            **Output Format:** A JSON list of up to **{current_num}** subtopics, e.g.:  

            '["Subtopic 1", "...", "Subtopic {current_num}"]'

            **Ensure that:**  
            - All subtopics can increase and decrease in magnitude, but **does not** contain increase or decrease or any causal relations 
            - Each subtopic is distinct and separate from others.  
        """

        messages = messages_init + [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        content = response.choices[0].message.content.strip()
        print(f"{'  ' * (level - 1)}  → LLM Response: {content[:200]}..." if len(content) > 200 else f"{'  ' * (level - 1)}  → LLM Response: {content}")
        
        subtopics = parse_json_response(content)
        print(f"{'  ' * (level - 1)}  → Generated {len(subtopics)} subtopics: {subtopics}")

        parent_dict[topic] = {}  # Initialize the topic as a dictionary

        # Add subtopics to the parent dictionary as keys with empty dictionaries as their values
        for subtopic in subtopics:
            if subtopic not in seen_nodes:
                seen_nodes.add(subtopic)
                parent_dict[topic][subtopic] = {}  # Add as an empty dictionary (no children)
                remaining_subtopics = [s for s in subtopics if s != subtopic]
                expand_topic(subtopic, parent_dict[topic], level + 1, remaining_subtopics, core_topic, other_core_topics)

        # Generate and append synonyms directly to the parent dictionary
        if with_synonyms>0:
            synonyms = get_synonyms(topic,num=round(with_synonyms*(current_num/num)))
            for synonym in synonyms:
                if synonym not in seen_nodes:
                    seen_nodes.add(synonym)
                    parent_dict[topic][synonym] = {}  # Add synonym as a key with an empty dictionary

    # Step 1: Generate hierarchy first (without synonyms)
    for topic in topics:
        if topic not in seen_nodes:
            seen_nodes.add(topic)
            remaining_topics = [s for s in topics if s != topic]
            expand_topic(topic, hierarchy, 1, remaining_topics, topic, remaining_topics)

    return hierarchy

def clean_strings(data):
    if isinstance(data, dict):
        return {clean_strings(key): clean_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_strings(item) for item in data]
    elif isinstance(data, str):
        return data.replace("increase", "").replace("decrease", "").replace("Increase", "").replace("Decrease", "")
    else:
        return data
    

def flatten_hierarchy_to_df(hierarchy, parent_labels=None, level=0):
    """
    Recursively flattens the hierarchical JSON into a list of rows.
    Each row contains the subtopic and the hierarchy of labels (categories) for each level.

    :param hierarchy: The hierarchical JSON structure (nested dictionary).
    :param parent_labels: A list of labels for each level of the hierarchy (used to track the hierarchical levels).
    :param level: The current depth level in the hierarchy.
    :return: DataFrame containing the flattened structure.
    """
    rows = []

    if parent_labels is None:
        parent_labels = []

    for topic, subtopics in hierarchy.items():
        # Create a row for the current topic
        row = {"topic": topic}
        
        # Assign the parent topic to itself at level 0
        row.update({f"category {i}": label for i, label in enumerate(parent_labels)})
        
        # Add the current topic to its own category
        row[f"category {level}"] = topic
        
        rows.append(row)

        # If the topic has subtopics, recurse into them
        if isinstance(subtopics, dict) and subtopics:
            # Call the function recursively for subtopics
            rows.extend(flatten_hierarchy_to_df(subtopics, parent_labels + [topic], level + 1))

    return rows

def hierarchy_to_df(hierarchy):
    """
    Converts hierarchical data (JSON-like structure) into a pandas DataFrame,
    where each row represents a subtopic and each category corresponds to a level of the hierarchy.

    :param hierarchy: The hierarchical JSON structure (nested dictionary).
    :return: DataFrame containing the flattened structure.
    """
    rows = flatten_hierarchy_to_df(hierarchy)
    df = pd.DataFrame(rows)
    return df

def add_noise_row(df, column_name, num_samples=3):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Sample unique non-null values from the column
    items = df[column_name].dropna().unique().tolist()
    sampled_items = random.sample(items, min(num_samples, len(items)))

    # Construct prompt
    prompt = (
        f"Given the following terms: {', '.join(sampled_items)}, "
        "return a phrase that is similar in style and length to the terms, "
        "and represents a possible underlying entity that can increase or decrease."
        "**This term should represent the overlap of the terms given and belong to all categories seen**"
    )

    # Call Groq
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    # Extract the phrase
    phrase = response.choices[0].message.content.strip()

    # Construct new row
    new_row = {col: (np.nan if col != column_name else phrase) for col in df.columns}

    # Append to DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


# ========================
# Config
# ========================

CONFIGS = [
    # (theme, max_sub, depth, add_noise)
    ("Energy_Ecosystems_and_Humans",         5, 3, 0.0),
    ("Energy_Ecosystems_and_Humans",         5, 3, 0.25),
    ("Energy_Ecosystems_and_Humans",         5, 3, 0.5),
    ("Energy_Ecosystems_and_Humans",         3, 5, 0.0),
    ("Energy_Ecosystems_and_Humans",         3, 5, 0.25),
    ("Energy_Ecosystems_and_Humans",         3, 5, 0.5),
    ("Offshore_energy_impacts_on_fisheries", 5, 3, 0.0),
    ("Offshore_energy_impacts_on_fisheries", 5, 3, 0.25),
    ("Offshore_energy_impacts_on_fisheries", 5, 3, 0.5),
    ("Offshore_energy_impacts_on_fisheries", 3, 5, 0.0),
    ("Offshore_energy_impacts_on_fisheries", 3, 5, 0.25),
    ("Offshore_energy_impacts_on_fisheries", 3, 5, 0.5),
]

T = 1.0
SYNONYMS = 0
BRANCHING = "random"


def generate_dataset(theme, max_sub, depth, add_noise, t=T, synonyms=SYNONYMS, branching=BRANCHING):
    add_noise = float(add_noise)
    os.makedirs('data/synthetic/generated_data', exist_ok=True)

    with open('data/synthetic/theme_keys.json', 'r') as f:
        data = json.load(f)

    top_level_topics = data[theme]
    file_name = f'data/synthetic/generated_data/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise0.0_{branching}.csv'

    print(f"\n{'='*60}")
    print(f"Theme: {theme} | depth={depth} | max_sub={max_sub} | noise={add_noise}")
    print(f"Output: {file_name}")
    print(f"{'='*60}")

    if not os.path.exists(file_name):
        print("Generating new hierarchy...")
        hierarchy = generate_hierarchy(
            top_level_topics, theme, depth=depth,
            temperature=t, num=max_sub, model=model,
            with_synonyms=synonyms, branching=branching,
        )
        hierarchy = clean_strings(hierarchy)

        json_file = file_name.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(hierarchy, f, indent=2)
        print(f"Saved JSON: {json_file}")

        df = hierarchy_to_df(hierarchy)
        df.to_csv(file_name, index=False, escapechar='\\')
        print(f"Saved CSV: {file_name}")
    else:
        print(f"Loading existing: {file_name}")
        df = pd.read_csv(file_name)
        print(f"Loaded {df.shape[0]} rows.")

    if add_noise > 0.0:
        noise_file = f'data/synthetic/generated_data/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'
        if not os.path.exists(noise_file):
            num_noise = int(add_noise * len(df))
            print(f"Adding {num_noise} noise rows...")
            for _ in tqdm(range(num_noise), desc="Generating noise"):
                df = add_noise_row(df, column_name="topic", num_samples=10)
            df.to_csv(noise_file, index=False, escapechar='\\')
            print(f"Saved noisy dataset: {noise_file}")
        else:
            print(f"Noisy dataset already exists: {noise_file}")


for theme, max_sub, depth, add_noise in CONFIGS:
    generate_dataset(theme, max_sub, depth, add_noise)
