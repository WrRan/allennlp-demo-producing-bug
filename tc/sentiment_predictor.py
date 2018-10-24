import logging

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

logger = logging.getLogger(__name__)


@Predictor.register("sentiment_classifier")
class SentimentPredictor(Predictor):
    def load_line(self, line: str) -> JsonDict:
        instance = {
            "text": line.rstrip()
        }
        return instance

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict["text"]
        instance = self._dataset_reader.text_to_instance(text=text)
        return instance
