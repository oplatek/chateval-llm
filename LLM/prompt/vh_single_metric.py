from prompt import Prompt, register_prompt, PositiveExamplePrompt
from dataclasses import dataclass
from prompt.examples import StaticExamples, RetrievedExamples


@register_prompt(name="comparable-no-egs")
@dataclass
class ComparibleAppropriatnessPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Evaluate the quality of the response with respect to the context with continuous number between 1 and 5.
The higher the score, the better quality .
The quality to be evaluated is appropriateness. 
The appropriateness score represents how well is the response appropriate given the preceding dialogue?
{positive_examples}
Now complete the following example by replying with a score for appropriateness.
Context: {dialogue_context}
Response: {response}
Appropriateness: """
    expected_reply_template: str = """{appropriateness}"""
    qualities = ("appropriateness",)

@register_prompt(name="comparable-no-egs-single_digit")
@dataclass
class ComparibleAppropriatnessPromptSingle(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Evaluate the quality of the response with respect to the context with continuous number between 1 and 5.
The higher the score, the better quality .
The quality to be evaluated is appropriateness. 
The appropriateness score represents how well is the response appropriate given the preceding dialogue?
{positive_examples}
Now complete the following example by replying with a score for appropriateness.
Context: {dialogue_context}
Response: {response}
Appropriateness: """
    expected_reply_template: str = """{appropriateness}"""
    qualities = ("appropriateness",)

@register_prompt(name="comparable-2-egs-dd-zhao")
@dataclass
class ComparibleAppropriatnessPrompt2egs(
    ComparibleAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = StaticExamples(
        positive_example_nested_template="""
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        num_examples=2,
        seed=42,
        dataset="dailydialog-zhao",
        qualities=("appropriateness",),
    )

@register_prompt(name="comparable-2-dyn-egs-dd-zhao")
@dataclass
class ComparibleAppropriatnessPrompt2DynEgs(
    ComparibleAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = RetrievedExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        dataset="dailydialog-zhao",
        vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        bk_vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        num_examples=2,
        seed=42,
        qualities=("appropriateness",),
    )


@register_prompt(name="comparable-4-dyn-egs-dd-zhao")
@dataclass
class ComparibleAppropriatnessPrompt2DynEgs(
    ComparibleAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = RetrievedExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        dataset="dailydialog-zhao",
        vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        bk_vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        num_examples=4,
        seed=42,
        qualities=("appropriateness",),
    )

@register_prompt(name="comparable-8-dyn-egs-dd-zhao")
@dataclass
class ComparibleAppropriatnessPrompt2DynEgs(
    ComparibleAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = RetrievedExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        dataset="dailydialog-zhao",
        vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        bk_vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        num_examples=8,
        seed=42,
        qualities=("appropriateness",),
    )

@register_prompt(name="op-appropriateness-single_digit-fasttext")
@dataclass
class OpAppropriatnessSingleDigitPrompt(Prompt):
    template: str = """
You are a helpful and precise assistant for checking the quality of the answer.


We will show you examples of the dialogues so you you will get familiar with the formatting.
Ignore the rating.

{positive_examples}
Similarly to the examples above, we will present you dialogue consisting of context and the final response.
Rate the appropriateness of the response for given context using digits: 1, 2, 3, 4, or 5.
The rating of 5 is for the best responses and rating of 1 is for the worst responses.

Return just a single digit after you read the Response for the Context below.
Context: {dialogue_context}
Response: {response}
Appropriateness: """
    expected_reply_template: str = """{appropriateness}"""
    qualities = ("appropriateness",)


@register_prompt(name="op-appropriateness-single_digit-fasttext-egs-dd-zhao-2-01")
@dataclass
class MlOpAppropriatnessPromptStaticEgs(
    OpAppropriatnessSingleDigitPrompt, PositiveExamplePrompt
):
    get_positive_examples = StaticExamples(
        positive_example_nested_template="""
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        num_examples=2,
        seed=42,
        dataset="dailydialog-zhao",
        qualities=("appropriateness",),
    )


@register_prompt(name="mlop-appropriateness-03")
@dataclass
class MlOpAppropriatnessPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it. Rate the appropriateness of the response for a given context by number in range from 1.1 to 4.9 
When the response fits well to the context (i.e. previous utterance), it should be higher. On the other hand, when the response is inappropriate for a given context, it should be lower. 
{positive_examples}
Now return the score preferably followed by your reasoning about the appropriateness of the Response for the Context below.
Context: {dialogue_context}
Response: {response}
Appropriateness: """
    expected_reply_template: str = """{appropriateness}"""
    qualities = ("appropriateness",)


@register_prompt(name="mlop-appropriateness-egs-dd-zhao-2-02")
@dataclass
class MlOpAppropriatnessPromptStaticEgs(
    MlOpAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = StaticExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        num_examples=2,
        seed=42,
        dataset="dailydialog-zhao",
        qualities=("appropriateness",),
    )

@register_prompt(name="mlop-appropriateness-dyn-egs-dd-zhao-2-02")
@dataclass
class MlOpAppropriatnessPromptStaticEgs(
    MlOpAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = RetrievedExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        dataset="dailydialog-zhao",
        vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        bk_vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        num_examples=2,
        seed=42,
        qualities=("appropriateness",),
    )


@register_prompt(name="mlop-appropriateness-egs-rh-copy-4-02")
@dataclass
class MlOpAppropriatnessPromptStaticEgs(
    MlOpAppropriatnessPrompt, PositiveExamplePrompt
):
    get_positive_examples = StaticExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness: {appropriateness}
        """,
        num_examples=4,
        seed=42,
        dataset="rehearsal-copy-hard",
        qualities=("appropriateness",),
    )


@register_prompt(name="ml-appropriateness-04")
@dataclass
class MlAppropriatnessPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it. Rate the appropriateness of the response for a given context on a scale between 1 and 5. 
When the response fits well to the context (i.e. previous utterance), it should receive 5. On the other hand, when the response is inappropriate for a given context, it should receive 1. 
Output just a single number.
Context: {dialogue_context}
Response: {response} 
Appropriateness: """
    expected_reply_template: str = """{appropriateness}"""
    qualities = ("appropriateness",)


@register_prompt(name="vh-appropriateness-03")
@dataclass
class VHAppropriatnessPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Express how the response is appropriate given context with continuous number between 1 and 5.
The higher the score, the more appropriate the sentences is.
Here's a few examples:
--------
{positive_examples}
--------
Now complete the following with just a single float number:
Context: {dialogue_context}
Response: {response}
Appropriateness Score: """
    expected_reply_template: str = """{appropriateness}"""
    qualities = ("appropriateness",)


@register_prompt(name="vh-appropriateness-2egs-dd-zhao-03")
class VHAppropriatnessPosExPrompt(VHAppropriatnessPrompt, PositiveExamplePrompt):
    get_positive_examples = StaticExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness Score: {appropriateness}
        """,
        num_examples=2,
        seed=42,
        dataset="dailydialog-zhao",
        qualities=("appropriateness",),
    )


@register_prompt(name="vh-appropriateness-2egs-dd-zhao-retrieval-01")
class VHApproriatnessPosExRetrPrompt(VHAppropriatnessPrompt, PositiveExamplePrompt):
    get_positive_examples = RetrievedExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Appropriateness Score: {appropriateness}
        """,
        dataset="dailydialog-zhao",
        vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        bk_vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        num_examples=2,
        seed=42,
        qualities=("appropriateness",),
    )


@register_prompt(name="vh-relevance-01")
@dataclass
class VHRelevancePrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Express how the response is relevant to the context with continuous number between 1 and 5.
The higher the score, the more relevant the sentences is.
Output just a single float number.
Here's a few examples:
--------
{positive_examples}
---------
Now complete the following:
Context: {dialogue_context}
Response: {response}
Relevance Score: """
    expected_reply_template: str = """{relevance}"""
    qualities = ("relevance",)


@register_prompt(name="vh-relevance-2egs-dd-zhao-02")
class VHRelevancePosExPrompt(VHRelevancePrompt, PositiveExamplePrompt):
    get_positive_examples = StaticExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Relevance Score: {relevance}
        """,
        num_examples=2, seed=42, dataset="dailydialog-zhao", qualities=("relevance",)
    )


@register_prompt(name="vh-relevance-2egs-dd-zhao-retrieval-01")
class VHRelevancePosExRetrPrompt(VHRelevancePrompt, PositiveExamplePrompt):
    get_positive_examples = RetrievedExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Relevance Score: {relevance}
        """,
        dataset="dailydialog-zhao",
        vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        bk_vectorstore_fn="vectorstores/dd_zhao-full-st.pkl",
        num_examples=2,
        seed=42,
        qualities=("relevance",),
    )


@register_prompt(name="vh-richness-02")
@dataclass
class VHRichnessPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Express how the response is rich with content or expressive continuous number between 1 and 5.
The higher the score, the more rich the sentences is.
Output just a single float number.
{positive_examples}
Now complete the following:
Context: {dialogue_context}
Response: {response}
Content Richness Score: """
    expected_reply_template: str = """{richness}"""
    qualities = ("richness",)


@register_prompt(name="vh-02-richness-2egs-dd-zhao")
class VHRichnessPosExPrompt(VHRichnessPrompt, PositiveExamplePrompt):
    get_positive_examples = StaticExamples(
        positive_example_nested_template="""
Context: {dialogue_context}
Response: {response}
Content Richness Score: {richness}
        """,
        num_examples=2, seed=42, dataset="dailydialog-zhao", qualities=("richness",)
    )


@register_prompt(name="vh-grammatical-02")
@dataclass
class VHGrammaticalPrompt(Prompt):
    template: str = """
Following is a dialogue context and the response to it.
Express how the response is grammatical and if it is semantically coherent with continuous number between 1 and 5.
The higher the score, the more grammatical the sentences is.
Output just a single float number.
{positive_examples}
Now complete the following:
Context: {dialogue_context}
Response: {response}
Grammar Score: """
    expected_reply_template: str = """{grammatical}"""
    qualities = ("grammatical",)


@register_prompt(name="vh-grammatical-2egs-dd-zhao-02")
class VHGrammaticalPosExPrompt(VHGrammaticalPrompt, PositiveExamplePrompt):
    get_positive_examples = StaticExamples(
        positive_example_nested_template = """
Context: {dialogue_context}
Response: {response}
Grammar Score: {grammatical}""",
        num_examples=2, seed=42, dataset="dailydialog-zhao", qualities=("grammatical",)
    )
