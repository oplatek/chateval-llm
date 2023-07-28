import pickle
import random
import logging

from chateval.results import TURN_LEVEL_METRICS
from chateval.datasets import load_dataset

class StaticExamples:
    def __init__(self, positive_example_nested_template, dataset, qualities, num_examples=2, seed=42):
        # It is CRITICAL to keep constructor/__init__ function ligh-weight since it is invoked during imports of Prompts classes 
        # since it is instantiated as class attribute.
        self.dataset = dataset
        self.qualities = qualities

        self.num_examples = num_examples
        self.seed = seed
        self.examples = None

        self.positive_example_nested_template = positive_example_nested_template
        self.positive_example_separator = "\n----------\n"
        self.main_positive_example_template = "\nPositive Examples (separated by newline and ---------- ):\n{positive_examples}\n\nThis was the last example.\n"

    def _prepare_examples(self):

        # TODO we might have mismatch in eval type but IMO it is not a big deal
        _df, context_list, response_list, gold_qualities = load_dataset(self.dataset, "en") 

        logging.info(f"Using the dataset {self.dataset} annotated with {gold_qualities.keys()} for few shot examples.")

        for q in self.qualities:
            assert f"annotations.{q}" in gold_qualities, f"Quality {q} not found in dataset! {gold_qualities.keys()}"
        
        assert len(context_list) == len(response_list) == len(gold_qualities[f"annotations.{self.qualities[0]}"])
        # TODO check seed is used correctly
        idxs = random.sample(list(range(len(context_list))), self.num_examples)
        context_list = [context_list[i] for i in idxs]
        response_list = [response_list[i] for i in idxs]
        gold_qualities = {f"annotations.{q}": [gold_qualities[f"annotations.{q}"][i] for i in idxs] for q in self.qualities}
        self.examples = [{"dialogue_context": c, "response": r} for c, r in zip(context_list, response_list)]
        for q, values in gold_qualities.items():
            for i, v in enumerate(values):
                assert q.startswith("annotations.")
                qstrip = q[len("annotations."):]
                self.examples[i][qstrip] = v

        logging.info(f"The examples are: {self('DummyDialogueContext', 'DummyResponse')}")

    def egs2str(self, examples_dicts):
        generated_egs = [self.positive_example_nested_template.format(**d) for d in examples_dicts]
        positive_examples = self.positive_example_separator.join(generated_egs)
        return self.main_positive_example_template.format(positive_examples=positive_examples)

    def __call__(self, dialogue_context, response):
        # static examples ignore the dialogue context and response
        if self.examples is None:
            self._prepare_examples()
        return self.egs2str(self.examples)



class RetrievedExamples(StaticExamples):
    def __init__(self, positive_example_nested_template, dataset, vectorstore_fn, bk_vectorstore_fn, qualities, num_examples=2, seed=42):
        # It is CRITICAL to keep constructor/__init__ function ligh-weight since it is invoked during imports of Prompts classes 
        # since it is instantiated as class attribute.
        super().__init__(positive_example_nested_template, dataset, qualities, num_examples, seed)
        self._initialized = False
        self.vectorstore = vectorstore_fn
        self.backup_vectorstore = bk_vectorstore_fn

    def _load_stores(self):
        self._initialized = True
        logging.info(f"Loading vectorstore from '{self.vectorstore}'")
        with open(self.vectorstore, "rb") as fd:
            self.vectorstore = pickle.load(fd)
        logging.info(f"Loading backup vectorstore from '{self.backup_vectorstore}'")
        with open(self.backup_vectorstore, "rb") as fd:
            self.backup_vectorstore = pickle.load(fd)
        logging.info(f"Using the dataset {self.dataset} and qualities '{self.qualities}' for few shot examples.")

    def _prepare_examples(self, db_key):

        # it seems that MMR doesnt work better
        # retrieved_examples = self.vectorstore.max_marginal_relevance_search(db_key, k=self.num_examples, fetch_k=10)
        retrieved_examples = self.vectorstore.similarity_search(db_key, k=self.num_examples * 3)
        retrieved_examples = [ex for ex in retrieved_examples if all((f"annotations.{q}" in ex.metadata["qualities"] for q in self.qualities))]
        retrieved_examples = retrieved_examples[:self.num_examples]
        if len(retrieved_examples) < self.num_examples:
            backup_examples = self.backup_vectorstore.similarity_search(db_key, k=self.num_examples-len(retrieved_examples))
            retrieved_examples.extend(backup_examples)

        examples = [{"dialogue_context": ret_example.metadata["context"],
                     "response": ret_example.metadata["response"]}
                    for ret_example in retrieved_examples]
        for q in self.qualities:
            for n, example in enumerate(examples):
                example[q] = retrieved_examples[n].metadata["qualities"][f"annotations.{q}"]

        return examples

    def __call__(self, dialogue_context, response):
        if not self._initialized:
            self._load_stores()
        # static examples ignore the dialogue context and response
        db_key = dialogue_context
        examples = self._prepare_examples(db_key)
        return self.egs2str(examples)

