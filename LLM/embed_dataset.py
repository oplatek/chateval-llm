import logging
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """Dataset for embeddings and raw targets score
    Storing examples in self.examples as a list of list of dicts
    The first list is for the ensemble, the second list is for the examples for given LLM model and its embeddings.
    The dict contains the following keys:
    """

    def __init__(self, paths, args, has_gt=True):
        self.args = args
        self.has_gt = has_gt
        assert isinstance(paths, list) and len(paths) > 0
        self.examples = []  # list of list of dicts
        for path in paths:
            map_location = torch.device("cpu") if args.gpus == 0 else None
            embeddings = torch.load(path, map_location=map_location)
            if has_gt:
                missed = [
                    x for x in embeddings if f"annotations.{self.quality}" not in x
                ]
                if len(missed) > 0:
                    logging.warning(
                        f"For {path}\n\tThe number of examples missing {self.quality}:\n\t{len(missed)}\n\tembeddings: {missed[:10]}\n\n"
                    )
                embeddings = [
                    x for x in embeddings if f"annotations.{self.quality}" in x
                ]
            self.examples.append(embeddings)
            # sanity check - compare with the examples for the first path
            for e1, e2 in zip(embeddings, self.examples[0]):
                assert (
                    e1["dialogue_id"] == e2["dialogue_id"]
                ), f"{e1['dialogue_id']} != {e2['dialogue_id']}"

    @property
    def quality(self):
        return self.args.quality

    def __len__(self):
        return len(self.examples[0])

    def __getitem__(self, idx):
        ensemble_embs = []
        for i in range(len(self.examples)):
            emb = self.examples[i][idx]["embedding"]
            # TODO - check if we need to convert to float16
            # emb = emb.float16() if self.args.precision == 16 else emb.float()
            if self.args.include_quality:
                q = torch.tensor([self.examples[i][idx][self.quality]])
                emb = torch.cat([emb, q])
            ensemble_embs.append(emb)
        emb = torch.cat(ensemble_embs)
        d = {
            f"dialogue_id": self.examples[0][idx]["dialogue_id"],
            "x": emb,
        }
        if self.has_gt:
            d[f"annotations.{self.quality}"] = self.examples[0][idx][
                f"annotations.{self.quality}"
            ]
        return d

    @property
    def input_size(self):
        return self.__getitem__(0)["x"].shape[0]
