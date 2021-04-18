import torch
from tqdm import tqdm
from config import Config
from transformers import BertTokenizer
from utils import DataIterator
import logging
import os
import numpy as np
import json
import pandas as pd

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list))
    return target_ts.to(device)


def find_neighbour(start, end, answer_len):
    answer = []
    for s_idx, s in enumerate(start):
        if s == 1:
            for e_idx, e in enumerate(end[s_idx:s_idx + answer_len]):
                if e == 1:
                    answer.append((s_idx, s_idx + e_idx))
                    break
    return answer


def refind_answer(test_rs_pd, id_list, start_prob_list, end_prob_list, questionlen_list,
                  allmaping_list, context_list, alltype_list):
    pred_answer_list = []
    C = 0
    answer_len = 64
    for id in tqdm(test_rs_pd['id']):
        score_dict = {}
        unique_index = [i for i, j in enumerate(id_list) if j == id]
        truepara_answer = 'SPECIL_TOKEN'
        for idx in unique_index:
            start_ = np.array(start_prob_list[idx])[questionlen_list[idx]:-1]  # context位置start概率
            end_ = np.array(end_prob_list[idx])[questionlen_list[idx]:-1]  # context位置start概率
            start_matrix = np.where(start_ >= 0.5, 1, 0)
            end_matrix = np.where(end_ >= 0.5, 1, 0)
            answer_ = find_neighbour(start_matrix, end_matrix, answer_len)  # 就近原则
            start_logits, end_logits, start_index, end_index = 0, 0, 0, 0
            for start_idx, end_idx in answer_:
                start_prob = start_[start_idx]  # 最大概率
                end_prob = end_[end_idx]  # 最大概率
                if start_prob + end_prob > start_logits + end_logits:
                    start_logits = start_prob
                    end_logits = start_prob
                    start_index = start_idx
                    end_index = end_idx
            try:
                real_start_index = allmaping_list[idx][start_index][0]
                real_end_index = allmaping_list[idx][end_index][-1]
            except:
                real_start_index, real_end_index = 0, 0
            if real_start_index >= real_end_index:
                continue
            if real_end_index - real_start_index + 1 > 100:
                continue
            answer = context_list[idx][real_start_index:real_end_index + 1]
            if answer == '':
                continue
            """打分"""
            w1, w2 = [0.98, 0.02]
            epsilon = 1e-3
            cls_prob = alltype_list[idx][0]
            start_cls = start_prob_list[idx][0]
            end_cls = end_prob_list[idx][0]
            pos_cls = -(start_cls * end_cls)
            answer_score = start_logits * end_logits - pos_cls
            score = np.exp((w1 * np.log(cls_prob + epsilon) + w2 * np.log(answer_score + epsilon)) / (
                    w1 + w2))
            score_dict[answer] = float(score)
            # score_dict[answer] = float(answer_score)
        try:
            """最高分"""
            answer = sorted(score_dict, key=score_dict.__getitem__, reverse=True)[0]
        except:
            try:
                answer = context_list[unique_index[0]][:answer_len]  # 分数最高的段落
            except:
                answer = ''
        if truepara_answer == answer:
            """最高分为实际候选集"""
            C += 1
        pred_answer_list.append(answer)
    return pred_answer_list, C


def set_test(test_iter, model_file, predict_df):
    model = torch.load(model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("***** Running Prediction *****")
    logger.info("  Predict Path = %s", model_file)
    model.eval()
    cls_pred = []
    start_pred = []
    end_pred = []
    all_uid_list, start_prob_list, end_prob_list, questionlen_list, allmaping_list, context_list, cls_prob_list = [], [], [], [], [], [], []
    for input_ids_list, input_mask_list, segment_ids_list, start_list, end_list, uid_list, \
        answer_list, text_list, querylen_list, mapping_list, cls_list in tqdm(
        test_iter):
        y_preds = model(list2ts2device(input_ids_list), list2ts2device(input_mask_list),
                        list2ts2device(segment_ids_list))
        y_maps = [torch.where(moid > 0.5, torch.ones_like(moid), torch.zeros_like(moid)) for moid in
                  y_preds]
        pred_s, pred_e, pred_c = (p.int().detach().cpu().tolist() for p in y_maps)
        prob_s, prob_e, prob_c = (p.detach().cpu().tolist() for p in y_preds)
        cls_pred.extend(pred_c)
        [start_pred.extend(t) for t in pred_s]
        [end_pred.extend(t) for t in pred_e]
        all_uid_list.extend(uid_list)
        start_prob_list.extend(prob_s)
        end_prob_list.extend(prob_e)
        questionlen_list.extend(querylen_list)
        allmaping_list.extend(mapping_list)
        context_list.extend(text_list)
        cls_prob_list.extend(pred_c)

    pred_answer, C = refind_answer(predict_df, all_uid_list, start_prob_list, end_prob_list,
                                   questionlen_list,
                                   allmaping_list,
                                   context_list, cls_prob_list)
    predict_df['answer'] = pred_answer
    predict_df.to_csv(config.processed_data + 'result_dev.csv')


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])

    test_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'test.csv', config=config, tokenizer=tokenizer)

    predict_file = config.processed_data + 'NCPPolicies_test.csv'
    print('Predicting {}..........'.format(str(predict_file)))

    test_df = pd.read_csv(predict_file, sep='\t', error_bad_lines=False)

    set_test(test_iter, config.checkpoint_path, test_df)
