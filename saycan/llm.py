"""
SayCan LLM Module - Language Model Integration for Task Planning.

This module provides integration with Large Language Models (LLMs) for task
planning and action scoring. Originally designed for GPT-3, now adapted for
local Ollama-based models.

The LLM provides:
- Few-shot prompting for task decomposition
- Action scoring based on task context
- Natural language to action mapping

Original SayCan Repository:
    https://github.com/google-research/google-research/tree/master/saycan

Reference:
    Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in
    Robotic Affordances. arXiv preprint arXiv:2204.01691.
"""

import ollama

# Ollama client configuration
client = ollama.Client(host='http://localhost:11434')
ENGINE = "llama3.2:1b"

from config import PICK_TARGETS, PLACE_TARGETS

# LLM Cache for repeated queries
LLM_CACHE = {}


def gpt3_call(engine=ENGINE, prompt="", max_tokens=128, temperature=0):
    """
    Call the LLM with caching for repeated queries.

    Args:
        engine: Model name to use
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text response
    """
    cache_id = (engine, prompt, max_tokens, temperature)
    if cache_id in LLM_CACHE:
        print('cache hit, returning cached response')
        return LLM_CACHE[cache_id]

    ollama_options = {}
    if max_tokens > 0:
        ollama_options['num_predict'] = max_tokens
    if temperature > 0:
        ollama_options['temperature'] = temperature

    response = client.generate(model=engine, prompt=prompt, options=ollama_options)
    generated_text = response['response']
    LLM_CACHE[cache_id] = generated_text
    return generated_text


def gpt3_scoring(query, options, engine=ENGINE, limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
    """
    Score action options using the LLM.

    Note: For local models without log probability access, this returns
    uniform scores. The actual discrimination comes from affordance scoring.

    Args:
        query: Prompt context for scoring
        options: List of action options to score
        engine: Model name
        limit_num_options: Limit number of options to score
        option_start: Prefix for options (unused)
        verbose: Print scoring details
        print_tokens: Print token details (unused)

    Returns:
        Tuple of (scores dict, empty response dict)
    """
    if limit_num_options:
        options = options[:limit_num_options]
    verbose and print("Scoring", len(options), "options with uniform LLM scores.")

    # Uniform scores since local models don't provide log probs
    uniform_logprob = 0.0
    scores = {option: uniform_logprob for option in options}

    if verbose:
        for i, (option, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
            print(score, "\t", option)
            if i >= 10:
                break

    return scores, {}


def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
    """
    Generate all possible pick-and-place action options.

    Args:
        pick_targets: Dict of pickable objects (uses PICK_TARGETS if None)
        place_targets: Dict of place targets (uses PLACE_TARGETS if None)
        options_in_api_form: If True, use API format; otherwise natural language
        termination_string: String to append for task completion

    Returns:
        List of action option strings
    """
    if not pick_targets:
        pick_targets = PICK_TARGETS
    if not place_targets:
        place_targets = PLACE_TARGETS

    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = f"robot.pick_and_place({pick}, {place})"
            else:
                option = f"Pick the {pick} and place it on the {place}."
            options.append(option)

    options.append(termination_string)
    print("Considering", len(options), "options")
    return options


# Termination string for task completion
termination_string = "done()"

# Few-shot prompt examples for task decomposition
gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""