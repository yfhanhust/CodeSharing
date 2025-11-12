import numpy as np 
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

eps_tol_fpr = 1e-4
eps_tol_recall = 5e-2

def load_qwen_response(path_to_validation_good,path_to_validation_mal):
    #### load validation dataset 
    with open(path_to_validation_good,'r') as f:
        file_set = f.readlines()

    file_hd = file_set[0].split('\t')
    question_good_val = []
    good_response_val = []
    for k in range(1,len(file_set)):
        records = file_set[k].split('\t')
        good_response_val.append([int(records[j]) for j in range(1,len(records))])

    question_good_val = [file_hd[j] for j in range(1,len(file_hd)-1)]
    good_response_val = np.array(good_response_val)

    with open(path_to_validation_mal,'r') as f:
        file_set = f.readlines()

    file_hd = file_set[0].split('\t')
    question_mal_val = []
    mal_response_val = []
    for k in range(1,len(file_set)):
        records = file_set[k].split('\t')
        mal_response_val.append([int(records[j]) for j in range(1,len(records))])

    question_mal_val = [file_hd[j] for j in range(1,len(file_hd)-1)]
    mal_response_val = np.array(mal_response_val)
    mal_response_val = mal_response_val[:,0:-1]
    good_response_val = good_response_val[:,0:-1]


    #### check if the questions have the same order
    count_same_val = 0
    for k in range(len(question_good_val)):
        if question_good_val[k] == question_mal_val[k]:
            count_same_val += 1
    
    if count_same_val == len(question_good_val):
        return good_response_val,mal_response_val,question_good_val
    else:
        print('Question orders are not consistent. \n')
        return -1


def question_map(question_list_val,question_list_test):
    nlen_test = len(question_list_test)
    idx_in_val = []
    for k in range(nlen_test):
        idx_in_val.append(np.where(np.array(question_list_val) == question_list_test[k])[0][0])

    return idx_in_val

def ll_computing(good_response,mal_response,fraction,ntop):
    ll_score = []
    nqa = good_response.shape[1] 
    ngood = good_response.shape[0]
    nmal = mal_response.shape[0] 

    ##### sampling
    mal_data_idx = np.array(range(0,nmal))
    good_data_idx = np.array(range(0,ngood))
    np.random.shuffle(mal_data_idx)
    np.random.shuffle(good_data_idx)
    good_train_num = int(ngood * fraction)
    mal_train_num = int(nmal * fraction)
    good_train_idx = good_data_idx[:good_train_num]
    good_test_idx = good_data_idx[good_train_num:]
    mal_train_idx = mal_data_idx[:mal_train_num]
    mal_test_idx = mal_data_idx[mal_train_num:]
    ##### computing likelihood scores for each graph query question using the training data 
    for k in range(nqa):
        ll = (float(np.sum(mal_response[mal_train_idx,k])) / float(mal_train_num)) / (1e-5 + float(np.sum(good_response[good_train_idx,k])) / float(good_train_num))
        ll_score.append(np.log(ll+1e-5))
    
    ll_ranking_list = np.argsort(-1*np.array(ll_score))

    good_response_sub = good_response[:,ll_ranking_list[:ntop]]
    mal_response_sub = mal_response[:,ll_ranking_list[:ntop]]
    
    benign_ll_train = np.zeros((good_response.shape[0],ntop))
    mal_ll_train = np.zeros((mal_response.shape[0],ntop))

    for k in range(ntop):
        for i in range(ngood):
            if good_response_sub[i][k] > 0:
                benign_ll_train[i][k] = ll_score[ll_ranking_list[k]]
        
        for i in range(nmal):
            if mal_response_sub[i][k] > 0:
                mal_ll_train[i][k] = ll_score[ll_ranking_list[k]]

    ##### applying the likelihood score table to the testing data
    benign_ll_test = np.zeros((len(good_test_idx),ntop))
    mal_ll_test = np.zeros((len(mal_test_idx),ntop))
    ngood_test = len(good_test_idx)
    nmal_test = len(mal_test_idx)
    for k in range(ntop):
        for i in range(ngood_test):
            if good_response_sub[good_test_idx[i]][k] > 0:
                benign_ll_test[i][k] = ll_score[ll_ranking_list[k]]
        
        for i in range(nmal_test):
            if mal_response_sub[mal_test_idx[i]][k] > 0:
                mal_ll_test[i][k] = ll_score[ll_ranking_list[k]]
    
    return [benign_ll_train, mal_ll_train, benign_ll_test, mal_ll_test]

    
def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0  # to avoid division by zero
    return 2 * (precision * recall) / (precision + recall)

def question_evaluation(good_response_list,mal_response_list):
    ll_score = []
    nqa = good_response_list.shape[1] 
    ngood = good_response_list.shape[0]
    nmal = mal_response_list.shape[0] 
    ##### computing likelihood scores for each graph query question. 
    for k in range(nqa):
        ll = (float(np.sum(mal_response_list[:,k])) / float(nmal)) / (1e-5 + float(np.sum(good_response_list[:,k])) / float(ngood))
        ll_score.append(np.log(ll+1e-5))

    ll_ranking_list = np.argsort(-1.*np.array(ll_score))

    return ll_ranking_list

def compute_metrics(y_true,y_pred):
    mean_fpr = np.linspace(0, 1, 1000)
    fpr,tpr,thresholds = roc_curve(y_true,y_pred)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    auc_score = roc_auc_score(y_true, y_pred)

    mean_fpr_diff = np.abs(mean_fpr - 5e-3)
    fnr_fpr005 = 1. - np.max(interp_tpr[np.where(mean_fpr_diff < eps_tol_fpr)[0]])  
    mean_fpr_diff = np.abs(mean_fpr - 1e-2)
    fnr_fpr01 = 1. - np.max(interp_tpr[np.where(mean_fpr_diff < eps_tol_fpr)[0]]) 
    mean_fpr_diff = np.abs(mean_fpr - 5e-2)
    fnr_fpr05 = 1. - np.max(interp_tpr[np.where(mean_fpr_diff < eps_tol_fpr)[0]]) 

    #### we report Precision, Recall and F1 score correspondingly 
    mean_recall = np.linspace(0,1,1000)
    precision, recall, _ = precision_recall_curve(y_true,y_pred)
    interp_precision = np.interp(mean_recall,recall,precision)
    interp_precision[0] = 0.0
    mean_recall_diff = np.abs(recall - 0.88)
    precision_recall088 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
    f1_score_recall088 = compute_f1(precision_recall088,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))
    mean_recall_diff = np.abs(recall - 0.80)
    precision_recall080 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
    f1_score_recall080 = compute_f1(precision_recall080,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))
    mean_recall_diff = np.abs(recall - 0.60)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        precision_recall060 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
        f1_score_recall060 = compute_f1(precision_recall080,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall060 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall060 = compute_f1(precision_recall080,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))



    mean_recall_diff = np.abs(recall - 0.50)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        if len(np.where(mean_recall_diff < 1e-1)[0]) == 0:
            precision_recall050 = 1. 
            f1_score_recall050 = compute_f1(precision_recall050,0.50)
        else:
            precision_recall050 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
            f1_score_recall050 = compute_f1(precision_recall050,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall050 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall050 = compute_f1(precision_recall050,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))

    mean_recall_diff = np.abs(recall - 0.30)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        if len(np.where(mean_recall_diff < 1e-1)[0]) == 0:
            precision_recall030 = 1. 
            f1_score_recall030 = compute_f1(precision_recall030,0.30)
        else:
            precision_recall030 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
            f1_score_recall030 = compute_f1(precision_recall030,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall030 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall030 = compute_f1(precision_recall030,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))

    mean_recall_diff = np.abs(recall - 0.20)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        if len(np.where(mean_recall_diff < 1e-1)[0]) == 0:
            precision_recall020 = 1.0
            f1_score_recall020 = compute_f1(precision_recall020,0.20)
        else:
            precision_recall020 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
            f1_score_recall020 = compute_f1(precision_recall020,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall020 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall020 = compute_f1(precision_recall030,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))

    return [auc_score, fnr_fpr005, fnr_fpr01,fnr_fpr05,precision_recall088,precision_recall080,precision_recall060,precision_recall050,precision_recall030,precision_recall020,f1_score_recall088,f1_score_recall080,f1_score_recall060,f1_score_recall050,f1_score_recall030,f1_score_recall020]

#### cross-validation
def graphqa_roc_nb(good_response_list,mal_response_list,fraction,ntop):
    ll_score = []
    nqa = good_response_list.shape[1] 
    ngood = good_response_list.shape[0]
    nmal = mal_response_list.shape[0]
    mal_data_idx = np.array(range(0,nmal))
    good_data_idx = np.array(range(0,ngood))
    np.random.shuffle(mal_data_idx)
    np.random.shuffle(good_data_idx)
    good_train_num = int(ngood * fraction)
    mal_train_num = int(nmal * fraction)
    good_train_idx = good_data_idx[:good_train_num]
    good_test_idx = good_data_idx[good_train_num:]
    mal_train_idx = mal_data_idx[:mal_train_num]
    mal_test_idx = mal_data_idx[mal_train_num:]
 
    ##### computing likelihood scores for each graph query question. 
    for k in range(nqa):
        ll = (float(np.sum(mal_response_list[mal_train_idx,k])) / float(mal_train_num)) / (1e-5 + float(np.sum(good_response_list[good_train_idx,k])) / float(good_train_num))
        ll_score.append(np.log(ll+1e-5))
    
    ll_ranking_list = np.argsort(-1*np.array(ll_score))
    ### testing 
    benign_ll_score = np.zeros(good_response_list[good_test_idx,:].shape[0])
    mal_ll_score = np.z##### computing likelihood scores for each graph query question. 
    for k in range(nqa):
        ll = (float(np.sum(mal_response_list[mal_train_idx,k])) / float(mal_train_num)) / (1e-5 + float(np.sum(good_response_list[good_train_idx,k])) / float(good_train_num))
        ll_score.append(np.log(ll+1e-5))
    
    ll_ranking_list = np.argsort(-1*np.array(ll_score))
    ### testing 
    benign_ll_score = np.zeros(good_response_list[good_test_idx,:].shape[0])
    mal_ll_score = np.zeros(mal_response_list[mal_test_idx,:].shape[0])

    for k in range(ntop):
        for i in range(len(good_test_idx)):
            if good_response_list[good_test_idx[i]][ll_ranking_list[k]] > 0:
                benign_ll_score[i] += ll_score[ll_ranking_list[k]]
        
        for i in range(len(mal_test_idx)):
            if mal_response_list[mal_test_idx[i]][ll_ranking_list[k]] > 0:
                mal_ll_score[i] += ll_score[ll_ranking_list[k]]

    
    y_true = len(mal_test_idx)*[1] + len(good_test_idx)*[0]
    y_pred = mal_ll_score.tolist() + benign_ll_score.tolist()eros(mal_response_list[mal_test_idx,:].shape[0])

    for k in range(ntop):
        for i in range(len(good_test_idx)):
            if good_response_list[good_test_idx[i]][ll_ranking_list[k]] > 0:
                benign_ll_score[i] += ll_score[ll_ranking_list[k]]
        
        for i in range(len(mal_test_idx)):
            if mal_response_list[mal_test_idx[i]][ll_ranking_list[k]] > 0:
                mal_ll_score[i] += ll_score[ll_ranking_list[k]]

    
    y_true = len(mal_test_idx)*[1] + len(good_test_idx)*[0]
    y_pred = mal_ll_score.tolist() + benign_ll_score.tolist()
    mean_fpr = np.linspace(0, 1, 1000)
    fpr,tpr,thresholds = roc_curve(y_true,y_pred)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    auc_score = roc_auc_score(y_true, y_pred)

    #### we fix FPR levels = 5e-3, 1e-2 and 5e-2, we report FNR correspondingly
    mean_fpr_diff = np.abs(mean_fpr - 5e-3)
    fnr_fpr005 = 1. - np.max(interp_tpr[np.where(mean_fpr_diff < eps_tol_fpr)[0]])  
    mean_fpr_diff = np.abs(mean_fpr - 1e-2)
    fnr_fpr01 = 1. - np.max(interp_tpr[np.where(mean_fpr_diff < eps_tol_fpr)[0]]) 
    mean_fpr_diff = np.abs(mean_fpr - 5e-2)
    fnr_fpr05 = 1. - np.max(interp_tpr[np.where(mean_fpr_diff < eps_tol_fpr)[0]]) 

    #### we report Precision, Recall and F1 score correspondingly 
    mean_recall = np.linspace(0,1,1000)
    precision, recall, _ = precision_recall_curve(y_true,y_pred)
    interp_precision = np.interp(mean_recall,recall,precision)
    interp_precision[0] = 0.0
    mean_recall_diff = np.abs(recall - 0.88)
    precision_recall088 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
    f1_score_recall088 = compute_f1(precision_recall088,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))
    mean_recall_diff = np.abs(recall - 0.80)
    precision_recall080 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
    f1_score_recall080 = compute_f1(precision_recall080,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))
    mean_recall_diff = np.abs(recall - 0.60)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        precision_recall060 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
        f1_score_recall060 = compute_f1(precision_recall080,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall060 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall060 = compute_f1(precision_recall080,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))


    mean_recall_diff = np.abs(recall - 0.50)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        if len(np.where(mean_recall_diff < 1e-1)[0]) == 0:
            precision_recall050 = 1. 
            f1_score_recall050 = compute_f1(precision_recall050,0.50)
        else:
            precision_recall050 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
            f1_score_recall050 = compute_f1(precision_recall050,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall050 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall050 = compute_f1(precision_recall050,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))

    mean_recall_diff = np.abs(recall - 0.30)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        if len(np.where(mean_recall_diff < 1e-1)[0]) == 0:
            precision_recall030 = 1. 
            f1_score_recall030 = compute_f1(precision_recall030,0.30)
        else:
            precision_recall030 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
            f1_score_recall030 = compute_f1(precision_recall030,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall030 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall030 = compute_f1(precision_recall030,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))

    mean_recall_diff = np.abs(recall - 0.20)
    if len(np.where(mean_recall_diff < eps_tol_recall)[0]) == 0:
        if len(np.where(mean_recall_diff < 1e-1)[0]) == 0:
            precision_recall020 = 1.0
            f1_score_recall020 = compute_f1(precision_recall020,0.20)
        else:
            precision_recall020 = np.mean(precision[np.where(mean_recall_diff < 1e-1)[0]])
            f1_score_recall020 = compute_f1(precision_recall020,np.mean(recall[np.where(mean_recall_diff < 1e-1)[0]]))
    else:
        precision_recall020 = np.mean(precision[np.where(mean_recall_diff < eps_tol_recall)[0]])
        f1_score_recall020 = compute_f1(precision_recall030,np.mean(recall[np.where(mean_recall_diff < eps_tol_recall)[0]]))

    
    return [auc_score, fnr_fpr005, fnr_fpr01,fnr_fpr05,precision_recall088,precision_recall080,precision_recall060,precision_recall050,precision_recall030,precision_recall020,f1_score_recall088,f1_score_recall080,f1_score_recall060,f1_score_recall050,f1_score_recall030,f1_score_recall020]


file_path = '/Mistral22_baseline/'
#### file paths to save responses over benignware / malware samples in the validation dataset 
path_to_validation_good = file_path + 'KG_query_likelihood_benignware_graph_query_prompt_response.csv'
path_to_validation_mal  = file_path + 'KG_query_likelihood_malware_graph_query_prompt_response.csv'

#### file paths to save responses over benignware / malware samples in the testing dataset 
path_to_test_good = file_path + 'KG_query_test_benignware_graph_query_prompt_response.csv'
path_to_test_mal  = file_path + 'KG_query_test_malware_graph_query_prompt_response.csv'

#### file paths to save the results of Exp 1 and Exp 2 (Cross-validation test) 
path_to_exp1_avg = file_path + 'cross-validation-top100-avg.csv' 
path_to_exp1_var = file_path + 'cross-validation-top100-var.csv' 
path_to_exp2_avg = file_path + 'cross-validation-05-fea-avg.csv'
path_to_exp2_var = file_path + 'cross-validation-05-fea-var.csv'
path_to_exp3_avg = file_path + 'test_3classifier_metrics_avg.csv'
path_to_exp3_var = file_path + 'test_3classifier_metrics_var.csv'



good_response_val, mal_response_val, question_list_val = load_qwen_response(path_to_validation_good,path_to_validation_mal)
good_response_test, mal_response_test, question_list_test = load_qwen_response(path_to_test_good,path_to_test_mal)




#### Exp.1 and 2 are done over the validation dataset. Only Exp 3 are conducted over the testing set. 
#### Exp.1 cross-validation with different franction levels of samples used for training using the top 62 questions
#### We report F1, AUC, FNR (FPR=1e-2, 5e-2), Precision, Recall
#train_frac = [0.3,0.4,0.5,0.6,0.7]
#test_frac = 1. - train_frac
#idx_test2val = question_map(question_list_val,question_list_test)

#good_response_val_sub = good_response_val[:,idx_test2val]
#mal_response_val_sub = mal_response_val[:,idx_test2val]
question_idx = question_map(question_list_val,question_list_test)

good_response_val_sub = good_response_val[:,question_idx]
mal_response_val_sub = mal_response_val[:,question_idx]

good_response_all = np.concatenate((good_response_val_sub,good_response_test))
mal_response_all = np.concatenate((mal_response_val_sub,mal_response_test))

nround = 100
result_metric_round = []
mean_fprs = []
interp_tprs = []
eps_tol_recall = 4e-2
for iround in range(nround):
    print(iround)
    result_metric_list = []
    for fraction in [0.3,0.4,0.5,0.6,0.7]:
        result_metrics = graphqa_roc_nb(good_response_all,mal_response_all,fraction,len(question_list_test))
        has_nan = np.isnan(result_metrics).any()
        if has_nan:
            continue
        else:
            result_metric_list.append(result_metrics)
    
    result_metric_round.append(result_metric_list)

#### Let's compute the averaged metrics across different rounds 
result_metric_round = np.array(result_metric_round)
averaged_metric = np.mean(result_metric_round,axis=0)
variance_metric = np.var(result_metric_round,axis=0)

fraction = [0.3,0.4,0.5,0.6,0.7]
names = ['auc_score','fnr_fpr005','fnr_fpr01','fnr_fpr05','precision_recall088','precision_recall080','precision_recall060','precision_recall050'
             ,'precision_recall030','precision_recall020','f1_score_recall088','f1_score_recall080','f1_score_recall060','f1_score_recall050','f1_score_recall030','f1_score_recall020']
df = pd.DataFrame(averaged_metric, index=fraction, columns=names)
df.index.name = 'training data fraction number'  # optional: name the index column
# Save to CSV
df.to_csv(path_to_exp1_avg)
            
df = pd.DataFrame(variance_metric, index=fraction, columns=names)
df.index.name = 'training data fraction number'  # optional: name the index column
# Save to CSV
df.to_csv(path_to_exp1_var)



#### Exp.2 cross-validation with diffrent numbers of query questions
#### fix train_frac = 0.5 
nround = 20 
result_metric_round_ntop = []
eps_tol_recall = 6e-2


for iround in range(nround):
    print(iround)
    result_metric_list_ntop = []
    for ntop in [30,40,50,60,62,70,80,90,100]:
        result_metrics = graphqa_roc_nb(good_response_all,mal_response_all,0.5,ntop)
        has_nan = np.isnan(result_metrics).any()
        if has_nan:
            continue
        else:
            result_metric_list_ntop.append(result_metrics)
        
    result_metric_round_ntop.append(result_metric_list_ntop)

result_metric_round_ntop = np.array(result_metric_round_ntop)
averaged_metric_ntop = np.mean(result_metric_round_ntop,axis=0)
variance_metric_ntop = np.var(result_metric_round_ntop,axis=0)



nfeature = [30,40,50,60,62,70,80,90,100]
names = ['auc_score','fnr_fpr005','fnr_fpr01','fnr_fpr05','precision_recall088','precision_recall080','precision_recall060','precision_recall050'
             ,'precision_recall030','precision_recall020','f1_score_recall088','f1_score_recall080','f1_score_recall060','f1_score_recall050','f1_score_recall030','f1_score_recall020']
df = pd.DataFrame(averaged_metric_ntop, index=nfeature, columns=names)
df.index.name = 'The number of selected questions'  # optional: name the index column
# Save to CSV
df.to_csv(path_to_exp2_avg)
            
df = pd.DataFrame(variance_metric_ntop, index=nfeature, columns=names)
df.index.name = 'training data fraction number'  # optional: name the index column
# Save to CSV
df.to_csv(path_to_exp2_var)

######### Exp.3 Test with RF, GBM and LR classifiers 
question_idx = question_map(question_list_val,question_list_test)
good_response_val_sub = good_response_val[:,question_idx]
mal_response_val_sub = mal_response_val[:,question_idx]

##### computing likelihood scores for each graph query question. 
nqa = 100
ll_score = []
for k in range(nqa):
    ll = (float(np.sum(mal_response_val_sub[:,k])) / float(mal_response_val_sub.shape[0])) / (1e-5 + float(np.sum(good_response_val_sub[:,k])) / float(good_response_val_sub.shape[0]))
    ll_score.append(np.log(ll+1e-5))

### training
num_good_train = good_response_val_sub.shape[0]
num_bad_train = mal_response_val_sub.shape[0]
benign_ll_mat_train = np.zeros((num_good_train,nqa))
mal_ll_mat_train = np.zeros((num_bad_train,nqa))
##### computing likelihood scores for each graph query question. 
for k in range(num_good_train):
    for i in range(nqa):
        if good_response_val_sub[k,i] > 0:
            benign_ll_mat_train[k,i] = ll_score[i]
        else:
            benign_ll_mat_train[k,i] = 0.

for k in range(num_bad_train):
    for i in range(nqa):
        if mal_response_val_sub[k,i] > 0:
            mal_ll_mat_train[k,i] = ll_score[i]
        else:
            mal_ll_mat_train[k,i] = 0.

train_mat = np.concatenate((benign_ll_mat_train,mal_ll_mat_train),axis=0)
train_label = np.array(num_good_train*[0] + num_bad_train*[1])

### testing 
num_good_test = good_response_test.shape[0]
num_bad_test = mal_response_test.shape[0]
benign_ll_mat_test = np.zeros((num_good_test,nqa))
mal_ll_mat_test = np.zeros((num_bad_test,nqa))
##### computing likelihood scores for each graph query question. 
for k in range(num_good_test):
    for i in range(nqa):
        if good_response_test[k,i] > 0:
            benign_ll_mat_test[k,i] = ll_score[i]
        else:
            benign_ll_mat_test[k,i] = 0.

for k in range(num_bad_test):
    for i in range(nqa):
        if mal_response_test[k,i] > 0:
            mal_ll_mat_test[k,i] = ll_score[i]
        else:
            mal_ll_mat_test[k,i] = 0.

test_mat = np.concatenate((benign_ll_mat_test,mal_ll_mat_test),axis=0)
test_label = np.array(num_good_test*[0] + num_bad_test*[1])

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

nfraction = [10,20,30,40,50,60,70,80,90,100]
acc_list = []
f1_list = []
precision_list = []
recall_list = []
roc_score_list = []
fpr_list = []
fnr_list = []

for nfea in nfraction:
    clf = RandomForestClassifier(n_estimators=500).fit(train_mat[:,:nfea],train_label)
    y_pred = clf.predict(test_mat[:,:nfea])
    y_prob = clf.predict_proba(test_mat[:,:nfea])
    acc_list.append(accuracy_score(test_label,y_pred))
    f1_list.append(f1_score(test_label,y_pred))
    precision_list.append(precision_score(test_label,y_pred))
    recall_list.append(recall_score(test_label,y_pred))
    roc_score_list.append(roc_auc_score(test_label,y_prob[:,1]))
    tn, fp, fn, tp = confusion_matrix(test_label, y_pred, labels=[0,1]).ravel()
    fpr_list.append(fp/(fp+tn))
    fnr_list.append(fn/(tp+fn))



# Create a DataFrame to store results
metrics_dict = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'FPR', 'FNR'],
}

# Add each feature number as a column
for i, nfea in enumerate(nfraction):
    metrics_dict[str(nfea)] = [
        acc_list[i],
        f1_list[i],
        precision_list[i],
        recall_list[i],
        roc_score_list[i],
        fpr_list[i],
        fnr_list[i]
    ]

df_metrics = pd.DataFrame(metrics_dict)

# Save to CSV file
df_metrics.to_csv(file_path + 'classification_metrics_RF.csv', index=False)

print("✅ Metrics saved successfully to 'classification_metrics_RF.csv'")
print(df_metrics)


acc_gbm_list = []
f1_gbm_list = []
precision_gbm_list = []
recall_gbm_list = []
roc_score_gbm_list = []
fpr_gbm_list = []
fnr_gbm_list = []

for nfea in nfraction:
    clf=GradientBoostingClassifier(n_estimators=200).fit(train_mat[:,:nfea],train_label)
    y_pred = clf.predict(test_mat[:,:nfea])
    y_prob = clf.predict_proba(test_mat[:,:nfea])
    acc_gbm_list.append(accuracy_score(test_label,y_pred))
    f1_gbm_list.append(f1_score(test_label,y_pred))
    precision_gbm_list.append(precision_score(test_label,y_pred))
    recall_gbm_list.append(recall_score(test_label,y_pred))
    roc_score_gbm_list.append(roc_auc_score(test_label,y_prob[:,1]))
    tn, fp, fn, tp = confusion_matrix(test_label, y_pred, labels=[0,1]).ravel()
    fpr_gbm_list.append(fp/(fp+tn))
    fnr_gbm_list.append(fn/(tp+fn))


# Create a DataFrame to store results
metrics_dict = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'FPR', 'FNR'],
}

# Add each feature number as a column
for i, nfea in enumerate(nfraction):
    metrics_dict[str(nfea)] = [
        acc_gbm_list[i],
        f1_gbm_list[i],
        precision_gbm_list[i],
        recall_gbm_list[i],
        roc_score_gbm_list[i],
        fpr_gbm_list[i],
        fnr_gbm_list[i]
    ]

df_metrics = pd.DataFrame(metrics_dict)

# Save to CSV file
df_metrics.to_csv(file_path + 'classification_metrics_GBM.csv', index=False)

print("✅ Metrics saved successfully to 'classification_metrics_GBM.csv'")
print(df_metrics)

acc_lr_list = []
f1_lr_list = []
precision_lr_list = []
recall_lr_list = []
roc_score_lr_list = []
fpr_lr_list = []
fnr_lr_list = []

for nfea in nfraction:
    clf=LogisticRegression(C=10).fit(train_mat[:,:nfea],train_label)
    y_pred = clf.predict(test_mat[:,:nfea])
    y_prob = clf.predict_proba(test_mat[:,:nfea])
    acc_lr_list.append(accuracy_score(test_label,y_pred))
    f1_lr_list.append(f1_score(test_label,y_pred))
    precision_lr_list.append(precision_score(test_label,y_pred))
    recall_lr_list.append(recall_score(test_label,y_pred))
    roc_score_lr_list.append(roc_auc_score(test_label,y_prob[:,1]))
    tn, fp, fn, tp = confusion_matrix(test_label, y_pred, labels=[0,1]).ravel()
    fpr_lr_list.append(fp/(fp+tn))
    fnr_lr_list.append(fn/(tp+fn))

# Create a DataFrame to store results
metrics_dict = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'FPR', 'FNR'],
}

# Add each feature number as a column
for i, nfea in enumerate(nfraction):
    metrics_dict[str(nfea)] = [
        acc_lr_list[i],
        f1_lr_list[i],
        precision_lr_list[i],
        recall_lr_list[i],
        roc_score_lr_list[i],
        fpr_lr_list[i],
        fnr_lr_list[i]
    ]

df_metrics = pd.DataFrame(metrics_dict)

# Save to CSV file
df_metrics.to_csv(file_path + 'classification_metrics_LR.csv', index=False)

print("✅ Metrics saved successfully to 'classification_metrics_LR.csv'")
print(df_metrics)





