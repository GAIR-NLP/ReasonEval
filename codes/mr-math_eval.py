import json
import re
import argparse
from sklearn.metrics import roc_curve, auc

def calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num):
    precision_positive = correct_true_num / total_true_num
    recall_positive = correct_true_num / (correct_true_num + total_false_num - correct_false_num)
    F1_score_positives = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)

    precision_negative = correct_false_num / total_false_num
    recall_negative = correct_false_num / (correct_false_num + total_true_num - correct_true_num)
    F1_score_negatives = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)

    macro_F1 = (F1_score_positives + F1_score_negatives) / 2
    return macro_F1


def get_results_for_invalid_errors():
    test_set_filepath = './dataset/mr-math_invalid_errors.json'
    test_dataset = []
    with open(test_set_filepath) as f:
        for line in f:
            test_dataset.append(json.loads(line))

    score_dict = {}
    evaluators = ['roscoe-sa', 'roscoe-ss', 'gpt3_5_turbo', 'gpt4', 'math-shepherd_mistral-7b', 'reasoneval_llama2-7b',
                  'reasoneval_wizardmath-7b-v1.0', 'reasoneval_mistral-7b', 'reasoneval_llemma-7b',
                  'reasoneval_abel-7b-002', 'reasoneval_wizardmath-7b-v1.1', 'reasoneval_llemma-34b']
    for evaluator in evaluators:
        score_filepath = './eval_results/mr-math_invalid_errors/' + evaluator + '_eval_results.json'
        score_results_record = []
        with open(score_filepath) as f:
            for line in f:
                score_results_record.append(json.loads(line))
        score_dict[evaluator] = score_results_record

    threshold = {'math-shepherd_mistral-7b': 0.5,
                 'reasoneval_llama2-7b': 0.5,
                 'reasoneval_wizardmath-7b-v1.0': 0.5,
                 'reasoneval_mistral-7b': 0.5,
                 'reasoneval_llemma-7b': 0.5,
                 'reasoneval_abel-7b-002': 0.5,
                 'reasoneval_wizardmath-7b-v1.1': 0.5,
                 'reasoneval_llemma-34b': 0.5,
                 'roscoe-sa': 0.025,
                 'roscoe-ss': 0.025}
    ### solution-level F1 score
    print('**************invalid errors*********solution level**********macro f1 score*******')
    for evaluator_name, score_results in score_dict.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
            for test_data, score_result in zip(test_dataset, score_results):
                if test_data['model_output_solution_correctness'] == 'correct':
                    total_true_num += 1
                    if score_result['correctness_pred'] == 'Correct' or score_result['correctness_pred'] == 'correct':
                        correct_true_num += 1
                elif test_data['model_output_solution_correctness'] == 'wrong':
                    total_false_num += 1
                    if score_result['correctness_pred'] == 'Wrong' or score_result['correctness_pred'] == 'wrong':
                        correct_false_num += 1
        else:
            for test_data, score_result in zip(test_dataset, score_results):
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    solution_level_score = min(score_result['scores'])
                elif evaluator_name in ['roscoe-sa', 'roscoe-ss']:
                    solution_level_score = score_result['scores']
                else:
                    new_score_list = [item[2] + item[1] for item in score_result['scores']]
                    solution_level_score = min(new_score_list)

                if test_data['model_output_solution_correctness'] == 'correct':
                    if solution_level_score > threshold[evaluator_name]:
                        correct_true_num += 1
                    total_true_num += 1
                if test_data['model_output_solution_correctness'] == 'wrong' and test_data[
                    'model_output_solution_first_error_step'] != 'N/A':
                    total_false_num += 1
                    if solution_level_score < threshold[evaluator_name]:
                        correct_false_num += 1
        macro_f1_score = calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num)
        print(f"{evaluator_name}: {format(macro_f1_score, '.3f')}")

        ### step-level F1 score

    print('**************invalid errors*********step level**********macro f1 score*******')
    for evaluator_name, score_results in score_dict.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['roscoe-sa', 'roscoe-ss']:
            continue
        else:
            for test_data, score_result in zip(test_dataset, score_results):
                gt = []
                if test_data['model_output_solution_correctness'] == 'correct':
                    gt = [1] * len(test_data['model_output_step_format'])
                else:
                    gt = [1] * (int(test_data['model_output_solution_first_error_step']) - 1) + [0] + [
                        'N/A'] * (len(test_data['model_output_step_format']) - int(
                        test_data['model_output_solution_first_error_step']))

                pred = []
                if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
                    if score_result['correctness_pred'] == 'Correct' or score_result['correctness_pred'] == 'correct':
                        pred = [1] * len(test_data['model_output_step_format'])
                    else:
                        step_pred = re.findall('(\d+)', score_result['error_step_pred'])
                        step_pred = step_pred[0] if len(step_pred) > 0 else ''
                        if step_pred != '' and int(step_pred) <= len(test_data['model_output_step_format']):
                            step_pred = int(step_pred)
                            pred = [1] * (step_pred - 1) + [0] + [
                                'N/A'] * (len(test_data['model_output_step_format']) - step_pred)
                        else:
                            ### rare case
                            pred = [1] * len(test_data['model_output_step_format'])
                    assert len(pred) == len(gt)

                else:
                    scores = []
                    if evaluator_name in ['math-shepherd_mistral-7b']:
                        raw_scores = score_result['scores']
                    else:
                        raw_scores = [item[2] + item[1] for item in score_result['scores']]
                    tag_len = 0
                    for step_idx, sub_step in enumerate(test_data['model_output_step_format']):
                        scores.append(min(raw_scores[tag_len:len(sub_step) + tag_len]))
                        tag_len += len(sub_step)
                    assert len(scores) == len(test_data['model_output_step_format'])
                    pred = [1 if score > threshold[evaluator_name] else 0 for score in scores]

                for gt_label, pred_label in zip(gt, pred):
                    if gt_label == 1:
                        total_true_num += 1
                    elif gt_label == 0:
                        total_false_num += 1
                    else:
                        continue
                    if gt_label == pred_label == 0:
                        correct_false_num += 1
                    if gt_label == pred_label == 1:
                        correct_true_num += 1
        macro_f1_score = calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num)
        print(f"{evaluator_name}: {format(macro_f1_score, '.3f')}")

    ### solution-level AUC
    print('**************invalid errors*********solution level**********AUC*******')
    for evaluator_name, score_results in score_dict.items():
        if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
            continue
        else:
            pred = []
            target = []
            for test_data, score_result in zip(test_dataset, score_results):
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    pred.append(min(score_result['scores']))
                elif evaluator_name in ['roscoe-sa', 'roscoe-ss']:
                    pred.append(score_result['scores'])
                else:
                    pred.append(min([item[2] + item[1] for item in score_result['scores']]))
                if test_data['model_output_solution_correctness'] == 'wrong' and test_data[
                    'model_output_solution_first_error_step'] != 'N/A':
                    target.append(0)
                else:
                    target.append(1)

            fpr, tpr, thresholds = roc_curve(target, pred)
            print(f"{evaluator_name}: {format(auc(fpr, tpr), '.3f')}")

    ### step-level AUC
    print('**************invalid errors*********step level**********AUC*******')
    for evaluator_name, score_results in score_dict.items():
        if evaluator_name in ['gpt3_5_turbo', 'gpt4', 'roscoe-sa', 'roscoe-ss']:
            continue
        else:
            pred = []
            target = []
            for test_data, score_result in zip(test_dataset, score_results):
                scores = []
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    raw_scores = score_result['scores']
                else:
                    raw_scores = [item[2] + item[1] for item in score_result['scores']]
                tag_len = 0
                for step_idx, sub_step in enumerate(test_data['model_output_step_format']):
                    scores.append(min(raw_scores[tag_len:len(sub_step) + tag_len]))
                    tag_len += len(sub_step)

                if test_data['model_output_solution_correctness'] == 'correct':
                    target.extend([1] * len(test_data['model_output_step_format']))
                    pred.extend(scores)
                if test_data['model_output_solution_correctness'] == 'wrong' and test_data[
                    'model_output_solution_first_error_step'] != 'N/A':
                    target.extend((test_data['model_output_solution_first_error_step'] - 1) * [1] + [0])
                    pred.extend(scores[:test_data['model_output_solution_first_error_step']])

            fpr, tpr, thresholds = roc_curve(target, pred)
            print(f"{evaluator_name}: {format(auc(fpr, tpr), '.3f')}")


def get_results_for_redundant_errors():
    test_set_filepath = './dataset/mr-math_redundant_errors.json'
    test_dataset = []
    with open(test_set_filepath) as f:
        for line in f:
            test_dataset.append(json.loads(line))

    score_dict = {}
    evaluators = ['roscoe-sa', 'roscoe-ss', 'gpt3_5_turbo', 'gpt4', 'math-shepherd_mistral-7b', 'reasoneval_llama2-7b',
                  'reasoneval_wizardmath-7b-v1.0', 'reasoneval_mistral-7b', 'reasoneval_llemma-7b',
                  'reasoneval_abel-7b-002', 'reasoneval_wizardmath-7b-v1.1', 'reasoneval_llemma-34b']
    for evaluator in evaluators:
        score_filepath = './eval_results/mr-math_redundant_errors/' + evaluator + '_eval_results.json'
        score_results_record = []
        with open(score_filepath) as f:
            for line in f:
                score_results_record.append(json.loads(line))
        score_dict[evaluator] = score_results_record

    threshold = {'math-shepherd_mistral-7b': 0.5,
                 'reasoneval_llama2-7b': -0.15,
                 'reasoneval_wizardmath-7b-v1.0': -0.15,
                 'reasoneval_mistral-7b': -0.15,
                 'reasoneval_llemma-7b': -0.15,
                 'reasoneval_abel-7b-002': -0.15,
                 'reasoneval_wizardmath-7b-v1.1': -0.15,
                 'reasoneval_llemma-34b': -0.15,
                 'roscoe-sa': 0.025,
                 'roscoe-ss': 0.025}

    ### solution-level F1 score
    print('**************redundant errors*********solution level**********macro f1 score*******')
    for evaluator_name, score_results in score_dict.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
            for test_data, score_result in zip(test_dataset, score_results):
                if 0 not in test_data['rating']:
                    total_true_num += 1
                    if score_result['correctness_pred'] == 'Absent' or score_result['correctness_pred'] == 'absent':
                        correct_true_num += 1
                elif 0 in test_data['rating']:
                    total_false_num += 1
                    if score_result['correctness_pred'] == 'Present' or score_result['correctness_pred'] == 'present':
                        correct_false_num += 1
        else:
            for test_data, score_result in zip(test_dataset, score_results):
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    solution_level_score = min(score_result['scores'])
                elif evaluator_name in ['roscoe-sa', 'roscoe-ss']:
                    solution_level_score = score_result['scores']
                else:
                    new_score_list = [-item[1] for item in score_result['scores']]
                    solution_level_score = min(new_score_list)

                if 0 not in test_data['rating']:
                    if solution_level_score > threshold[evaluator_name]:
                        correct_true_num += 1
                    total_true_num += 1
                if 0 in test_data['rating']:
                    total_false_num += 1
                    if solution_level_score < threshold[evaluator_name]:
                        correct_false_num += 1
        macro_f1_score = calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num)
        print(f"{evaluator_name}: {format(macro_f1_score, '.3f')}")

        ### step-level F1 score

    print('**************redundant errors*********step level**********macro f1 score*******')
    for evaluator_name, score_results in score_dict.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['roscoe-sa', 'roscoe-ss']:
            continue
        else:
            for test_data, score_result in zip(test_dataset, score_results):
                gt = test_data['rating']
                pred = []
                if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
                    all_ground_truth_1 = []
                    all_ground_truth_0 = []

                    for idx, rating in enumerate(gt):
                        if rating == 0:
                            all_ground_truth_0.append(idx + 1)
                        elif rating == 1:
                            all_ground_truth_1.append(idx + 1)

                    step_pred = re.findall('(\d+)', score_result['error_step_pred'])

                    all_record = []
                    for pred_loc in step_pred:
                        all_record.append(int(pred_loc))

                    for i in all_ground_truth_1:
                        total_true_num += 1
                        if i not in all_record:
                            correct_true_num += 1

                    for i in all_ground_truth_0:
                        total_false_num += 1
                        if i in all_record:
                            correct_false_num += 1

                else:
                    scores = []
                    if evaluator_name in ['math-shepherd_mistral-7b']:
                        raw_scores = score_result['scores']
                    else:
                        raw_scores = [-item[1] for item in score_result['scores']]
                    scores = raw_scores
                    assert len(scores) == len(test_data['model_output_step_format'])
                    pred = [1 if score > threshold[evaluator_name] else 0 for score in scores]

                    for gt_label, pred_label in zip(gt, pred):
                        if gt_label == 1:
                            total_true_num += 1
                        elif gt_label == 0:
                            total_false_num += 1
                        else:
                            continue
                        if gt_label == pred_label == 0:
                            correct_false_num += 1
                        if gt_label == pred_label == 1:
                            correct_true_num += 1
        macro_f1_score = calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num)
        print(f"{evaluator_name}: {format(macro_f1_score, '.3f')}")

    ### solution-level AUC
    print('**************redundant errors*********solution level**********AUC*******')
    for evaluator_name, score_results in score_dict.items():
        if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
            continue
        else:
            pred = []
            target = []
            for test_data, score_result in zip(test_dataset, score_results):
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    pred.append(min(score_result['scores']))
                elif evaluator_name in ['roscoe-sa', 'roscoe-ss']:
                    pred.append(score_result['scores'])
                else:
                    pred.append(min([-item[1] for item in score_result['scores']]))
                if 0 in test_data['rating']:
                    target.append(0)
                else:
                    target.append(1)

            fpr, tpr, thresholds = roc_curve(target, pred)
            print(f"{evaluator_name}: {format(auc(fpr, tpr), '.3f')}")

    ### step-level AUC
    print('**************redundant errors*********step level**********AUC*******')
    for evaluator_name, score_results in score_dict.items():
        if evaluator_name in ['gpt3_5_turbo', 'gpt4', 'roscoe-sa', 'roscoe-ss']:
            continue
        else:
            pred = []
            target = []
            for i,j in zip(test_dataset,score_results):
                scores = []
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    raw_scores = j['scores']
                else:
                    raw_scores = [- item[1] for item in j['scores']]
                scores = raw_scores
                target.extend(i['rating'])
                pred.extend(scores)
  
            fpr, tpr, thresholds = roc_curve(target,pred)
            print(f"{evaluator_name}: {format(auc(fpr, tpr),'.3f')}")


def get_results(error_type):
    if error_type == 'invalid':
        get_results_for_invalid_errors()
    else:
        get_results_for_redundant_errors()


if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_type', type=str, choices=['invalid', 'redundant'], default='invalid')
    args = parser.parse_args()
    get_results(args.error_type)

    # get_results('invalid')
    # get_results('redundant')
