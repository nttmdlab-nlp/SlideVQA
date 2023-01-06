import sys
import json
import numpy as np
import os
import string 
from collections import Counter
import re

WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

def normalize_answer(s, question):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    def yesno(text):
        if 'yes' == text[:3] or 'no' == text[:2]:
            text = text.split()[0]
        return text
    def replace_text(text):
        return text.replace('this is ', '').replace('it is ', '').replace('&', ',').replace('and', ',').replace('percent', '').replace('organisation', 'organization').replace('because of', '').replace('because', '').replace('due to', '').replace('hours', 'hrs').replace('minites', 'min')
    def word2number(text):
        words = text.split()
        return ' '.join([str(WORD_NUMBER_MAP[word]) if word in WORD_NUMBER_MAP else word for word in words])
    def remove_unit(text, question):
        if 'how many' in question:
            idx = question.find('how many')
            unit = question[idx+len('how many'):].split()[0]
            text = text.replace(unit, '')
        if 'which' in question:
            idx = question.find('which')
            unit = question[idx+len('which'):].split()[0]
            text = text.replace(unit, '')
        return text
    return word2number(white_space_fix(yesno(remove_articles(remove_punc(remove_unit(replace_text(lower(s)), question))))))

def evaluate_f1_em_qa(gts, preds):
    f1 = exact_match = 0
    precisions = {}
    recalls = {}
    ems = {}
    for qa_id in gts:
        question = gts[qa_id]['question']
        prediction = preds[qa_id]['answer']
        ground_truth = gts[qa_id]['answer']
        prediction_tokens = normalize_answer(prediction, question).split()
        ground_truth_tokens = normalize_answer(ground_truth, question).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            precisions[qa_id] = recalls[qa_id] = ems[qa_id] = 0
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 += (2 * precision * recall) / (precision + recall)
        exact_match += (prediction_tokens == ground_truth_tokens)
        precisions[qa_id] = precision
        recalls[qa_id] = recall
        ems[qa_id] = (prediction_tokens == ground_truth_tokens)
    exact_match = exact_match / len(gts)
    f1 = f1 / len(gts)
    return {'F1': f1, 'EM': exact_match, 'precisions': precisions, 'recalls': recalls, 'EMs': ems}

def evaluate_f1_em_es(gts, preds):
    f1 = exact_match = 0
    precisions = {}
    recalls = {}
    ems = {}
    for qa_id in gts:
        prediction = preds[qa_id]['evidence_pages']
        ground_truth = gts[qa_id]['evidence_pages']
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            precisions[qa_id] = recalls[qa_id] = ems[qa_id] = 0
            continue
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 += (2 * precision * recall) / (precision + recall)
        exact_match += (prediction == ground_truth)
        precisions[qa_id] = precision
        recalls[qa_id] = recall
        ems[qa_id] = (prediction == ground_truth)
    exact_match = exact_match / len(gts)
    f1 = f1 / len(gts)
    return {'F1': f1, 'EM': exact_match, 'precisions': precisions, 'recalls': recalls, 'EMs': ems}

def evaluate_f1_em_main(metrics_qa, metrics_es):
    qa_preicsions = metrics_qa['precisions']
    qa_recalls = metrics_qa['recalls']
    qa_ems = metrics_qa['EMs']
    es_preicsions = metrics_es['precisions']
    es_recalls = metrics_es['recalls']
    es_ems = metrics_es['EMs']
    f1 = exact_match = 0
    for qa_id in qa_preicsions:
        qa_preicsion = qa_preicsions[qa_id]
        qa_recall = qa_recalls[qa_id]
        qa_em = qa_ems[qa_id]
        es_preicsion = es_preicsions[qa_id]
        es_recall = es_recalls[qa_id]
        es_em = es_ems[qa_id]
        joint_precision = qa_preicsion * es_preicsion
        joint_recall = qa_recall * es_recall
        if (joint_precision + joint_recall) == 0:
            continue
        f1 += (2 * joint_precision * joint_recall) / (joint_precision + joint_recall)
        exact_match += int(qa_em == es_em == 1)
    f1 = f1 / len(qa_preicsions)
    exact_match = exact_match / len(qa_preicsions)
    return {'F1': f1, 'EM': exact_match}

def print_metrics(res_metrics, task):
    keys = ['F1', 'EM']
    print(f'**********\nFinal model performance ({task}):\n**********')
    for k in keys:
        print(k, ': %.1f' % (res_metrics[k] * 100))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_preds_file', type=str, default='qa_preds.jsonl')
    parser.add_argument('--es_preds_file', type=str, default='es_pred.jsonl')
    parser.add_argument('--gts_file', type=str, default='test.jsonl')
    args = parser.parse_args()

    with open(args.qa_preds_file) as f:
        qa_preds = f.read().splitlines()

    with open(args.es_preds_file) as f:
        es_preds = f.read().splitlines()

    with open(args.gts_file) as f:
        gts = f.read().splitlines()

    gts = {}
    preds = {}
    for qa_pred, es_pred, gt in zip(qa_preds, es_preds, gts):
        qa_pred = json.loads(qa_pred)
        es_pred = json.loads(es_pred)
        gt = json.loads(gt)
        qa_id = gt['qa_id']
        preds[qa_id] = {'question': pred['question'], 'answer': qa_pred['answer'], 'evidence_pages': es_pred['evidence_pages']}    
        gts[qa_id] = {'question': gt['question'], 'answer': gt['answer'], 'evidence_pages': gt['evidence_pages']}

    qa_metrics = evaluate_f1_em_qa(gts, preds)
    es_metrics = evaluate_f1_em_es(gts, preds)
    main_metrics = evaluate_f1_main(qa_metrics, es_metrics)

    print_metrics(qa_metrics, 'QA')
    print_metrics(es_metrics, 'ES')
    print_metrics(main_metrics, 'Main')
