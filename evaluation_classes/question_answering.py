import re

from evaluation.eval_base_class import Eval
from mdqa.MusiQue_metrics.answer import AnswerMetric
import numpy as np


class QA(Eval):

    def __init__(self, id_key, predictions_dir, out_path):
        self.correlations = []
        super().__init__(id_key, predictions_dir, out_path)

    def _evaluate(self, predictions, model_name, sample_index):
        answer_metric = AnswerMetric()
        follow_format = []
        for sample in predictions:
            gt = sample["target"]
            pred = self.postprocess(sample)
            follow_format.append(type(pred) is dict)
            # gt, pred = self.parse(gt, pred)
            answer_metric(predicted_answer=str(pred), ground_truth_answers=[str(gt)])

        metric = answer_metric.get_metric()
        self.correlations.append([np.mean(follow_format), np.mean(metric[3])])

        metrics = {"all_f1": metric[3]}
        return metrics

    def parse(self, ground_truth_answer, predicted_answer):
        gt = self._extract_answer_from_dict(ground_truth_answer)
        if type(predicted_answer) is dict:
            pred = self._extract_answer_from_dict(predicted_answer)
        else:
            pred = predicted_answer
        return gt, pred

    @staticmethod
    def _extract_answer_from_dict(answer_dict):
        if answer_dict["is_answerable"]:
            answer = answer_dict["answer_content"]
        else:
            answer = answer_dict["is_answerable"]
        return answer

    def postprocess(self, pred):
        only_response = self.get_only_response(pred)
        answer_dict = re.search(r'\{.*\}', only_response)
        if answer_dict is None:
            return only_response

        str_dict = answer_dict.group(0)
        str_dict = str_dict.replace("'s", "\\'s").replace("'t", "\\'t").replace("s' ", "s\\' ")
        str_dict = str_dict.replace("\\\\_", "_").replace("\\_", "_")
        try:
            answer = eval(str_dict)
        except Exception as e:
            try:
                str_dict = str_dict.replace("}", "'}")
                answer = eval(str_dict)
            except Exception as e:
                answer = only_response
        return answer


