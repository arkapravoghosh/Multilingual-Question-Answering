import json
import os

import datasets

_CITATION = """\
@inproceedings{Kumar2022IndicNLGSM,
  title={IndicNLG Suite: Multilingual Datasets for Diverse NLG Tasks in Indic Languages},
  author={Aman Kumar and Himani Shrotriya and Prachi Sahu and Raj Dabre and Ratish Puduppully and Anoop Kunchukuttan and Amogh Mishra and Mitesh M. Khapra and Pratyush Kumar},
  year={2022},
  url = "https://arxiv.org/abs/2203.05437"
}
"""

_DESCRIPTION = """\
This is the Question Generation dataset released as part of IndicNLG Suite. Each
example has five fields: id, squad_id, answer, context and question. We create this dataset in eleven
languages including as, bn, gu, hi, kn, ml, mr, or, pa, ta, te. This is a translated data. The examples in each language are exactly similar but in different languages.
The number of examples in each language is 98,027.
"""
_HOMEPAGE = "https://indicnlp.ai4bharat.org/indicnlg-suite"

_LICENSE = "Creative Commons Attribution-NonCommercial 4.0 International Public License"

_URL = "https://huggingface.co/datasets/ai4bharat/IndicQuestionGeneration/resolve/main/data/{}_IndicQuestionGeneration_v{}.tar.bz2"


_LANGUAGES = [
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te"
]


class WikiBio(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="{}".format(lang),
            version=datasets.Version("1.0.0")
        )
        for lang in _LANGUAGES
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            version=self.VERSION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang = str(self.config.name)
        url = _URL.format(lang, self.VERSION.version_str[:-2])

        data_dir = dl_manager.download_and_extract(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir,  lang + "_train" + ".jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_test" + ".jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_val" + ".jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            for idx_, row in enumerate(f):
                data = json.loads(row)
                idx = data["context"].find(data["answer"])
                # yield idx_, {
                #     "id": data["id"],
                #     "squad_id": data["squad_id"],
                #     "answer": data["answer"],
                #     "context": data["context"],
                #     "question": data["question"]

                # }

                # Update Dataset to HuggingFace SQUAD format
                yield idx_, {
                    "id": data["id"],
                    "title": data["squad_id"],
                    "answers":  {
                        "answer_start" : [idx],
                        "text" : [data["answer"]]
                    },
                    "context": data["context"],
                    "question": data["question"]
                }
