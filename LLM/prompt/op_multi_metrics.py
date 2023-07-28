from prompt import Prompt, register_prompt, PositiveExamplePrompt
from dataclasses import dataclass
from prompt.examples import StaticExamples, RetrievedExamples


@register_prompt(name="op-all-02")
@dataclass
class OPAppropriatnessPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Evaluate qualities of the response with respect to the context with continuous number between 1 and 5.
For each quality, the higher the score, the better quality .
The qualities to be evaluated are:
 - appropriateness:  The number represents how well is the response appropriate given the preceding dialogue?
 - richness: The number represents how much is the response informative, does it contain long sentences including multiple entities, conceptual and emotional words?
 - grammatical: The number measures if the responses are free of grammatical and semantic errors?
 -relevance: The number measures if the responses are on-topic with the immediate dialog history?
{positive_examples}
Now complete the following example. For each quality, repeat the quality name followed by the decimal number. 
Context: {dialogue_context}
Response: {response}
Qualities:
"""
    expected_reply_template: str = """
appropriateness: {appropriateness}, richness: {richness}, grammatical: {grammatical}, relevance: {relevance}
"""
    qualities = (
        "appropriateness",
        "richness",
        "grammatical",
        "relevance",
    )


@register_prompt(name="op-all-egs-rehearsal-copy-hard5-02")
@dataclass
class VHAppropriatnessPosExPrompt(OPAppropriatnessPrompt, PositiveExamplePrompt):
       get_positive_examples = StaticExamples(
        positive_example_nested_template="""
Context: {dialogue_context}
Response: {response}
Qualities:
  appropriateness: {appropriateness}
  richness: {richness}
  grammatical: {grammatical}
  relevance: {relevance}
        """,
        num_examples=5,
        seed=42,
        dataset="rehearsal-copy-hard",
        qualities=(
            "appropriateness",
            "richness",
            "grammatical",
            "relevance",
        ),
    )
