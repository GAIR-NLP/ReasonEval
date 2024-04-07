import json
import re
from sklearn.metrics import roc_curve, auc

def calculate_f1_score(correct_true_num, total_true_num, total_false_num, correct_false_num):
    precision_positive = correct_true_num / total_true_num
    recall_true_positive = correct_true_num / (correct_true_num + total_false_num - correct_false_num)
    F1_score_true_positives = 2 * precision_positive * recall_true_positive / (precision_positive + recall_true_positive)

    precision_negative = correct_false_num / total_false_num
    recall_true_negative = correct_false_num / (correct_false_num + total_true_num - correct_true_num)
    F1_score_true_negatives = 2 * precision_negative * recall_true_negative / (precision_negative + recall_true_negative)

    macro_F1_score = (F1_score_true_positives + F1_score_true_negatives) / 2
    return macro_F1_score


def get_results():
    test_set_filepath = './dataset/mr-gsm8k.json'
    test_dataset = []
    with open(test_set_filepath) as f:
        for line in f:
            test_dataset.append(json.loads(line))

    evaluator_scores = {}
    evaluator_names = ['roscoe-sa', 'roscoe-ss', 'gpt3_5_turbo', 'gpt4', 'math-shepherd_mistral-7b',
                       'reasoneval_llama2-7b', 'reasoneval_wizardmath-7b-v1.0', 'reasoneval_mistral-7b',
                       'reasoneval_llemma-7b', 'reasoneval_abel-7b-002', 'reasoneval_wizardmath-7b-v1.1',
                       'reasoneval_llemma-34b']
    for name in evaluator_names:
        score_filepath = './eval_results/mr-gsm8k/' + name + '_eval_results.json'
        score_results = []
        with open(score_filepath) as f:
            for line in f:
                score_results.append(json.loads(line))
            evaluator_scores[name] = score_results

    evaluator_thresholds = {'math-shepherd_mistral-7b': 0.5,
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
    for evaluator_name, score_results in evaluator_scores.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['gpt3_5_turbo']:
            for i, j in zip(test_dataset, score_results):
                if i['model_output_solution_correctness'] == 'correct':
                    total_true_num += 1
                    if j['gpt3_5_eval_output']['correctness_pred'] == 'Correct' or j['gpt3_5_eval_output'][
                        'correctness_pred'] == 'correct':
                        correct_true_num += 1
                elif i['model_output_solution_correctness'] == 'wrong':
                    total_false_num += 1
                    if j['gpt3_5_eval_output']['correctness_pred'] == 'Wrong' or j['gpt3_5_eval_output'][
                        'correctness_pred'] == 'wrong':
                        correct_false_num += 1
        elif evaluator_name in ['gpt4']:
            for i, j in zip(test_dataset, score_results):
                if i['model_output_solution_correctness'] == 'correct':
                    total_true_num += 1
                    if j['gpt4_eval_output']['correctness_pred'] == 'Correct' or j['gpt4_eval_output'][
                        'correctness_pred'] == 'correct':
                        correct_true_num += 1
                elif i['model_output_solution_correctness'] == 'wrong':
                    total_false_num += 1
                    if j['gpt4_eval_output']['correctness_pred'] == 'Wrong' or j['gpt4_eval_output'][
                        'correctness_pred'] == 'wrong':
                        correct_false_num += 1
        else:
            for i, j in zip(test_dataset, score_results):
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    solution_level_score = min(j['scores'])
                elif evaluator_name in ['roscoe-sa', 'roscoe-ss']:
                    solution_level_score = j['scores']
                else:
                    new_score_list = [item[2] + item[1] for item in j['scores']]
                    solution_level_score = min(new_score_list)

                if i['model_output_solution_correctness'] == 'correct':
                    if solution_level_score > evaluator_thresholds[evaluator_name]:
                        correct_true_num += 1
                    total_true_num += 1
                if i['model_output_solution_correctness'] == 'wrong' and i['model_output_solution_first_error_step'] != 'N/A':
                    total_false_num += 1
                    if solution_level_score < evaluator_thresholds[evaluator_name]:
                        correct_false_num += 1
        macro_F1_score_solution = calculate_f1_score(correct_true_num, total_true_num, total_false_num,
                                                     correct_false_num)
        print(f"{evaluator_name}: {format(macro_F1_score_solution, '.3f')}")

        ### step-level F1 score

    print('**************invalid errors*********step level**********macro f1 score*******')
    for evaluator_name, score_results in evaluator_scores.items():
        correct_true_num = 0
        correct_false_num = 0
        total_false_num = 0
        total_true_num = 0
        if evaluator_name in ['roscoe-sa', 'roscoe-ss']:
            continue
        else:
            for i, j in zip(test_dataset, score_results):
                gt = []
                if i['model_output_solution_correctness'] == 'correct':
                    gt = [1] * len(i['model_output_steps'])
                else:
                    gt = [1] * (int(i['model_output_solution_first_error_step']) - 1) + [0] + [
                        'N/A'] * (len(i['model_output_steps']) - int(i['model_output_solution_first_error_step']))

                pred = []
                if evaluator_name in ['gpt3_5_turbo']:
                    if j['gpt3_5_eval_output']['correctness_pred'] == 'Correct' or j['gpt3_5_eval_output'][
                        'correctness_pred'] == 'correct':
                        pred = [1] * len(i['model_output_steps'])
                    else:
                        step_pred = re.findall('(\d+)', j['gpt3_5_eval_output']['error_step_pred'])
                        step_pred = step_pred[0] if len(step_pred) > 0 else ''
                        if step_pred != '' and int(step_pred) <= len(i['model_output_steps']):
                            step_pred = int(step_pred)
                            pred = [1] * (step_pred - 1) + [0] + [
                                'N/A'] * (len(i['model_output_steps']) - step_pred)
                        else:
                            ### rare case
                            pred = [1] * len(i['model_output_steps'])
                    assert len(pred) == len(gt)

                elif evaluator_name in ['gpt4']:
                    if j['gpt4_eval_output']['correctness_pred'] == 'Correct' or j['gpt4_eval_output'][
                        'correctness_pred'] == 'correct':
                        pred = [1] * len(i['model_output_steps'])
                    else:
                        step_pred = re.findall('(\d+)', j['gpt4_eval_output']['error_step_pred'])
                        step_pred = step_pred[0] if len(step_pred) > 0 else ''
                        if step_pred != '' and int(step_pred) <= len(i['model_output_steps']):
                            step_pred = int(step_pred)
                            pred = [1] * (step_pred - 1) + [0] + [
                                'N/A'] * (len(i['model_output_steps']) - step_pred)
                        else:
                            ### rare case
                            pred = [1] * len(i['model_output_steps'])
                    assert len(pred) == len(gt)

                else:
                    scores = []
                    if evaluator_name in ['math-shepherd_mistral-7b']:
                        raw_scores = j['scores']
                    else:
                        raw_scores = [item[2] + item[1] for item in j['scores']]
                    scores = raw_scores
                    assert len(scores) == len(i['model_output_steps'])
                    pred = [1 if score > evaluator_thresholds[evaluator_name] else 0 for score in scores]

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
        macro_F1_score_step = calculate_f1_score(correct_true_num, total_true_num, total_false_num,
                                                 correct_false_num)
        print(f"{evaluator_name}: {format(macro_F1_score_step, '.3f')}")

    ### solution-level AUC
    print('**************invalid errors*********solution level**********AUC*******')
    for evaluator_name, score_results in evaluator_scores.items():
        if evaluator_name in ['gpt3_5_turbo', 'gpt4']:
            continue
        else:
            predictions = []
            target = []
            for i, j in zip(test_dataset, score_results):
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    predictions.append(min(j['scores']))
                elif evaluator_name in ['roscoe-sa', 'roscoe-ss']:
                    predictions.append(j['scores'])
                else:
                    predictions.append(min([item[2] + item[1] for item in j['scores']]))
                if i['model_output_solution_correctness'] == 'wrong' and i[
                    'model_output_solution_first_error_step'] != 'N/A':
                    target.append(0)
                else:
                    target.append(1)

            fpr, tpr, thresholds = roc_curve(target, predictions)
            print(f"{evaluator_name}: {format(auc(fpr, tpr), '.3f')}")

    ### step-level AUC
    print('**************invalid errors*********step level**********AUC*******')
    for evaluator_name, score_results in evaluator_scores.items():
        if evaluator_name in ['gpt3_5_turbo', 'gpt4', 'roscoe-sa', 'roscoe-ss']:
            continue
        else:
            predictions = []
            target = []
            for i, j in zip(test_dataset, score_results):
                scores = []
                if evaluator_name in ['math-shepherd_mistral-7b']:
                    raw_scores = j['scores']
                else:
                    raw_scores = [item[2] + item[1] for item in j['scores']]
                scores = raw_scores

                if i['model_output_solution_correctness'] == 'correct':
                    target.extend([1] * len(i['model_output_steps']))
                    predictions.extend(scores)
                if i['model_output_solution_correctness'] == 'wrong' and i[
                    'model_output_solution_first_error_step'] != 'N/A':
                    target.extend((i['model_output_solution_first_error_step'] - 1) * [1] + [0])
                    predictions.extend(scores[:i['model_output_solution_first_error_step']])

            fpr, tpr, thresholds = roc_curve(target, predictions)
            print(f"{evaluator_name}: {format(auc(fpr, tpr), '.3f')}")


if __name__ == '__main__':
    get_results()