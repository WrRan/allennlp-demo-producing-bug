import logging
from typing import Dict, Optional

from allennlp.common.checks import check_dimensions_match
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

import torch
from torch.nn import Dropout
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


@Model.register("cnn_classifier")
class CnnClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.aggregator = aggregator
        self.classifier_feedforward = classifier_feedforward
        if dropout > 0:
            self._dropout = Dropout(dropout)
        else:
            self._dropout = lambda x: x

        check_dimensions_match(self.text_field_embedder.get_output_dim(),
                               self.aggregator.get_input_dim(),
                               "text_field_embedder output_dim",
                               "aggregator input_dim")
        check_dimensions_match(self.aggregator.get_output_dim(),
                               self.classifier_feedforward.get_input_dim(),
                               "aggregator output_dim",
                               "classifier_feedforward input_dim")

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, text_len, embedded_dim)
        embedded_text = self._dropout(self.text_field_embedder(text))
        batch_size = embedded_text.size(0)

        # Shape: (batch_size, encoded_dim)
        encoded_text = self._dropout(self.aggregator(embedded_text, mask))

        # the final forward layer
        # Shape: (batch_size, 3)
        logits = self.classifier_feedforward(encoded_text)
        # Shape:
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, "probs": probs}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
