import logging
import sys
from typing import Any, Dict, List, Tuple, Callable
from dataclasses import dataclass
from chateval.results import TURN_LEVEL_METRICS

import re

registered_prompts = {}


def get_prompt(name: str, *args, **kwargs) -> Any:
    global registered_prompts
    cls = registered_prompts[name]
    return cls(*args, **kwargs)


class register_prompt:
    def __init__(self, name: str):
        global registered_prompts
        assert isinstance(name, str), "Please provide a str name for your prompt"
        assert (
            name not in registered_prompts
        ), f"Prompt with name {name} already registered. Choose different name"
        self.name = name

    def __call__(self, cls):
        global registered_prompts
        if cls is None:
            raise ValueError(
                "Please call register_prompt(name='YOUR_NAME') as decorator on your class."
            )
        registered_prompts[self.name] = cls
        cls.name = self.name
        logging.debug(f"Registered prompt: {self.name}")
        return cls


@dataclass
@register_prompt(name="base-simple-prompt")
class Prompt:
    qualities = TURN_LEVEL_METRICS  # OVERWRITE if your prompt template supports fewer qualities!
    # the template should have placeholders {dialog_context} and {response}
    template: str = """
Definition:

Evaluate the following Response in the given Dialogue Context with a floating number on scale of 1 to 5, where 1 indicates low quality and 5 indicates high quality. For each of the following qualities, please provide a separate score.
appropriateness;  How well is the response appropriate given the preceding dialogue?
richness; How mush is the response informative, does contain long sentences including multiple entities, conceptual and emotional words?
grammatical; Are the responses free of grammatical and semantic errors?
relevance; Are the responses are on-topic with the immediate dialog history?
For each quality, repeat the quality name followed by the decimal number. 
    Now complete the following example by evaluating the dialogue according the four qualities.

Input: 

Dialogue Context: 

{dialogue_context}

Response:

{response}

Qualities:
    """
    # the expected reply template should have placeholders for each metric
    expected_reply_template: str = """
appropriateness: {appropriateness}
richness: {richness}
grammatical: {grammatical}
relevant: {relevant}
    """
    default_missing_value: float = (
        3.0  # the default value to use if a metric is missing in the reply
    )
    _test_dialogue_context = "This is an EXAMPLE dialogue context\nwith several turns\neach reply is on a new line\ncapito?"
    _test_response = "This is an EXAMPLE response"

    def __post_init__(self):
        """Runs after initialization, the dataclass - think about it as an __init_() or as a constructor."""
        # safery checks
        assert self.name is not None
        assert self.template is not None
        for q in self.qualities:
            assert (
                q in TURN_LEVEL_METRICS
            ), f"{q} not in {TURN_LEVEL_METRICS}: {self.qualities}"
        # testing the template
        self(
            dialogue_context=self._test_dialogue_context, response=self._test_response
        )  # should not raise an error
        self.expected_reply_template.format(
            **{m: self.default_missing_value for m in self.qualities}
        )  # should not raise an error

    def parse_reply(
        self, reply: str, raise_on_missing=False, default_missing_value=None
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Override for your needs but check that all self.qualities are present in the reply.

        This method parses any string with format which contains "quality: float_value" pairs and
        returns a dictionary with the quality as key and the value as value.

        Since we expect for each metric the [1, 5] range the uninformed guess is 3.
        """
        if len(self.qualities) == 1:
            # Try to parse the reply as a single number
            try:
                value = float(reply.split()[0])
                return {self.qualities[0]: value}, {self.qualities[0]: 0}
            except Exception:
                logging.debug(
                    "For single quality template, returning only the number is best but parsing the reply failed"
                )

        # Parsinng the reply as a string with quality: value pairs
        default_missing_value = default_missing_value or self.default_missing_value
        words = re.split("[:;|,!? \t\n]", reply.lower())

        d, errors = {}, {q: 0 for q in self.qualities}
        for quality in self.qualities:
            value = None
            try:
                idx = words.index(quality)
            except ValueError:
                continue  # quality not found
            # searching for value following the quality key words
            for next_word in words[idx + 1 :]:
                if next_word in TURN_LEVEL_METRICS:
                    # the value would be relevant for another metric
                    break
                try:
                    value = float(next_word)
                    d[quality] = value
                    break  # we found the value
                except:
                    pass
        for quality in self.qualities:
            if quality not in d:
                errors[quality] += 1
                if raise_on_missing:
                    raise ValueError(
                        f"Quality {quality} not found in reply: {reply}. The expected reply template is: {self.expected_reply_template}"
                    )
                else:
                    d[quality] = default_missing_value
        return d, errors

    def __call__(self, dialogue_context: str, response: str) -> str:
        """Override if you want to do something fancy like generate positive and negative examples"""
        return self.template.format(
            dialogue_context=dialogue_context, response=response, positive_examples=""
        )


@register_prompt(name="base-positive-examples-prompt")
@dataclass
class PositiveExamplePrompt(Prompt):
    main_positive_example_template: str = """

Positive Examples:

{positive_examples}

    """
    positive_example_nested_template: str = ""
    positive_example_separator = """


    """
    get_postive_examples: Callable = None

    def __post_init__(self):
        super().__post_init__()
        return self(
            dialogue_context=self._test_dialogue_context, response=self._test_response
        )  # should not raise an error

    def __call__(self, dialogue_context: str, response: str) -> str:
        # The dictionary should have the following keys: dialoguecontext, response, and all the qualities
        positive_examples = self.get_positive_examples(dialogue_context, response)
        return self.template.format(
            dialogue_context=dialogue_context,
            response=response,
            positive_examples=positive_examples,
        )
