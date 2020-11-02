"""
# 使用nltk库实现BLEU评估算法
"""
import os
import logging
import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)


class BLEUEvaluator(object):
    def __init__(self,
                 predict_json_path='./online_test_data/test_answers.json',
                 target_json_path='./online_test_data/test_answers_target.json',
                 question_json_path=None,
                 good_qa_threshold=None):
        with open(predict_json_path, mode='r', encoding='utf-8') as file_predict:
            self.predict_content = json.load(file_predict)
        with open(target_json_path, mode='r', encoding='utf-8') as file_target:
            self.target_content = json.load(file_target)

        self.good_qa_threshold = None
        if question_json_path and good_qa_threshold:
            # 二者条件都满足才进行此项操作
            self.good_qa_threshold = good_qa_threshold
            with open(question_json_path, mode='r', encoding='utf-8') as file_question:
                # 获取问题列表
                self.questions_dict = self.get_question_dict(json.load(file_question))

        # 记录bleu分比较高的回答, 追溯其问题以及上下文
        self.good_test_questions = []
        self.bad_test_questions = []

    def eval(self):
        """评估函数"""
        predict_answers_dict = self.get_answer_dict(self.predict_content)
        target_answers_dict = self.get_answer_dict(self.target_content)

        return self.compute_bleu(predict_answers_dict, target_answers_dict)

    def compute_bleu(self, predict_dict, target_dict):
        """
        Args:
            predict_dict: 预测字典列表
            target_dict:

        Returns:

        """
        n_sum = 0
        smooth = SmoothingFunction()

        predict_data_length = len(predict_dict.keys())
        _logger.info("all predict data size: {}.".format(predict_data_length))
        for single_key in predict_dict.keys():
            if not target_dict.get(single_key):
                # 跳过查不大不到目标的数据
                predict_dict -= 1
                continue

            target_list_three = target_dict.get(single_key).split("<sep>")
            n_eval_result = sentence_bleu(target_list_three, predict_dict.get(single_key),
                                          smoothing_function=smooth.method1)

            print(n_eval_result)
            if self.good_qa_threshold:
                if n_eval_result > self.good_qa_threshold:
                    self.good_test_questions.append(self.good_qa_track(single_key, predict_dict.get(single_key), n_eval_result))
                else:
                    self.bad_test_questions.append(self.good_qa_track(single_key, predict_dict.get(single_key), n_eval_result))

            n_sum += n_eval_result

        _logger.info("resize predict data size: {}.".format(predict_data_length))

        return float(n_sum) / predict_data_length

    def good_qa_track(self, predict_single_key, predict_single_value, n_eval_result):
        """获取最佳QA对"""
        # 1. 对应ID的question context
        single_question = self.questions_dict.get(predict_single_key)
        # 2. 添加预测结果
        single_question['PredictAnswer'] = predict_single_value
        # 3. BLEU score
        single_question['BLEU'] = n_eval_result

        return single_question

    def get_answer_dict(self, answers_dict_list):
        """将dict list 转为dict, Id为key, answer为value"""
        answers_dict = {}
        for single_dict in answers_dict_list:
            answers_dict[single_dict.get('Id')] = single_dict.get('Answer')

        return answers_dict

    def get_question_dict(self, questions_dict_list):
        """将dict list 转为dict, Id为key, 整体dict为value"""
        questions_dict = {}
        for single_dict in questions_dict_list:
            questions_dict[single_dict.get('Id')] = single_dict

        return questions_dict

    def save_good_qa_question(self, file_path):
        """保存最佳qa的question"""
        with open(file_path, mode='w', encoding='utf-8') as fw:
            json.dump(self.good_test_questions, fw, ensure_ascii=False, indent=2)

    def save_bad_qa_question(self, file_path):
        """保存最佳qa的question"""
        with open(file_path, mode='w', encoding='utf-8') as fw:
            json.dump(self.bad_test_questions, fw, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of the sentence generation effect.")

    parser.add_argument('-p', '--predict_json_path',
                        default='../online_test_data/test_answers.json',
                        type=str,
                        help='The json file for the predict results.')
    parser.add_argument('-t', '--target_json_path',
                        default='../online_test_data/test_answers_target.json',
                        type=str,
                        help='The json file for the target results.')
    parser.add_argument('-q', '--question_json_path',
                        default='../online_test_data/test_questions.json',
                        type=str,
                        help='The json file for the question contents.')
    parser.add_argument('-gq', '--good_question_json_path',
                        default='../online_test_data/good_test_questions.json',
                        type=str,
                        help='The json file for the save good ga track question contents.')
    parser.add_argument('-bq', '--bad_question_json_path',
                        default='../online_test_data/bad_test_questions.json',
                        type=str,
                        help='The json file for the save bad ga track question contents.')
    parser.add_argument('-trd', '--good_qa_threshold',
                        default=None,
                        type=float,
                        help='the judge threshold for the predict and question is good qa track.')

    args = parser.parse_args()

    evaluator = BLEUEvaluator(predict_json_path=args.predict_json_path,
                              target_json_path=args.target_json_path,
                              question_json_path=args.question_json_path,
                              good_qa_threshold=args.good_qa_threshold)

    eval_result = evaluator.eval()
    _logger.info("eval result is {}.".format(eval_result))

    if args.good_qa_threshold:
        _logger.info("save good qa track question contents.")
        evaluator.save_good_qa_question(args.good_question_json_path)
        evaluator.save_bad_qa_question(args.bad_question_json_path)
