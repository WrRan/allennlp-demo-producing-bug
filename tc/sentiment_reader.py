import codecs
import logging
from typing import Iterable, Dict

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("sentiment_reader")
class SentimentReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 limit_n_token: int=300,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._limit_n_token = limit_n_token
        self.max_n_token_text = 0
        self.cnt_n_token_text = []

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with codecs.open(file_path, encoding="utf8") as data_file:
            for text in data_file:
                text = text.rstrip("\n\r")
                label = data_file.readline().rstrip("\n\r")
                yield self.text_to_instance(text, label)

    def text_to_instance(self,
                         text: str,
                         label: str = None) -> Instance:
        if label == "":
            label = None
        tokenized_text = self._tokenizer.tokenize(text)
        self.max_n_token_text = max(self.max_n_token_text, len(tokenized_text))
        self.cnt_n_token_text.append(len(tokenized_text))
        tokenized_text = tokenized_text[:self._limit_n_token]

        fields = {"text": TextField(tokenized_text, self._token_indexers)}
        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)
