import pickle
import numpy as np 
from sklearn.metrics import roc_curve, auc

file_folder = '/graph_query_prompt_response_benignware.csv'
with open(file_folder,'r') as f:
    file_set = f.readlines()

file_hd = file_set[0].split('\t')
file_hash = []
good_response_list = []
for k in range(1,len(file_set)):
    records = file_set[k].split('\t')
    file_hash.append(records[0])
    good_response_list.append([int(records[j]) for j in range(1,len(records))])

question_list_good = [file_hd[j] for j in range(1,len(file_hd)-1)]
good_response_list = np.array(good_response_list)


file_folder = '/graph_query_prompt_response_malware.csv'
with open(file_folder,'r') as f:
    file_set = f.readlines()

file_hd = file_set[0].split('\t')
file_hash = []
mal_response_list = []
for k in range(1,len(file_set)):
    records = file_set[k].split('\t')
    file_hash.append(records[0])
    mal_response_list.append([int(records[j]) for j in range(1,len(records))])

question_list_mal = [file_hd[j] for j in range(1,len(file_hd)-1)]
mal_response_list = np.array(mal_response_list)


### check if the two question lists are the same
count_same = 0
for k in range(len(question_list_good)):
    if question_list_good[k] == question_list_mal[k]:
        count_same += 1

print(count_same)
print(count_same == len(question_list_good))

mal_response_list = mal_response_list[:,0:-1]
good_response_list_all = good_response_list[:,0:-1]
nmal = mal_response_list.shape[0]
nben = good_response_list_all.shape[0]
nqa = len(question_list_good)
ll_score = []
for k in range(nqa):
    ll = (float(np.sum(mal_response_list[:,k])) / float(nmal)) / (1e-5 + float(np.sum(good_response_list_all[:,k])) / float(nben))
    print('mal prob: %f', np.sum(mal_response_list[:,k]) / float(nmal)) 
    print("benign prob: %f", np.sum(good_response_list_all[:,k])/float(nben))
    ll_score.append(np.log(ll+1e-5))

ll_ranking_list = np.argsort(-1.*np.array(ll_score))

benign_ll_score = np.zeros(good_response_list_all.shape[0])
mal_ll_score = np.zeros(mal_response_list.shape[0])

for k in range(good_response_list_all.shape[0]):
    for i in range(good_response_list_all.shape[1]):
        if good_response_list_all[k][i] > 0:
            benign_ll_score[k] += ll_score[i]

for k in range(mal_response_list.shape[0]):
    for i in range(mal_response_list.shape[1]):
        if mal_response_list[k][i] > 0:
            mal_ll_score[k] += ll_score[i]

y_true = mal_response_list.shape[0]*[1] + good_response_list_all.shape[0]*[0]
y_pred = mal_ll_score.tolist() + benign_ll_score.tolist()

from sklearn.metrics import f1_score, roc_auc_score, roc_curve
print(roc_auc_score(y_true, y_pred))
fpr,tpr,thresholds = roc_curve(y_true,y_pred)
idx = np.where(np.abs(fpr-0.01) < 0.005)[0]
print(fpr[idx])
print(tpr[idx])
print(thresholds[idx])
print(np.where(benign_ll_score > 14.60847303)[0].shape)
print(np.where(mal_ll_score < 14.60847303)[0].shape)

def graphqa_roc(good_response_list,mal_response_list,fraction,topk):
    #### select 200 training samples for likelihood score computation
    ll_score = []
    nqa = good_response_list.shape[1]
    ngood = good_response_list.shape[0]
    nmal = mal_response_list.shape[0]
    mal_data_idx = np.array(range(0,nmal))
    good_data_idx = np.array(range(0,ngood))
    np.random.shuffle(mal_data_idx)
    np.random.shuffle(good_data_idx)
    good_train_num = int(ngood * fraction)
    #good_test_num = ngood - good_train_num
    mal_train_num = int(nmal * fraction)
    #mal_test_num = nmal - mal_train_num
    good_train_idx = good_data_idx[:good_train_num]
    good_test_idx = good_data_idx[good_train_num:]
    mal_train_idx = mal_data_idx[:mal_train_num]
    mal_test_idx = mal_data_idx[mal_train_num:]
 
    for k in range(nqa):
        ll = (float(np.sum(mal_response_list[mal_train_idx,k])) / float(mal_train_num)) / (1e-5 + float(np.sum(good_response_list[good_train_idx,k])) / float(good_train_num))
        ll_score.append(np.log(ll+1e-5))
    
    ll_ranking_list = np.argsort(-1*np.array(ll_score))
    ### testing 
    benign_ll_score = np.zeros(good_response_list[good_test_idx,:].shape[0])
    mal_ll_score = np.zeros(mal_response_list[mal_test_idx,:].shape[0])

    for k in range(len(good_test_idx)):
        for i in range(topk):
            if good_response_list[good_test_idx[k]][ll_ranking_list[i]] > 0:
                benign_ll_score[k] += ll_score[ll_ranking_list[i]]

    for k in range(len(mal_test_idx)):
        for i in range(topk):
            if mal_response_list[mal_test_idx[k]][ll_ranking_list[i]] > 0:
                mal_ll_score[k] += ll_score[ll_ranking_list[i]]

    y_true = len(mal_test_idx)*[1] + len(good_test_idx)*[0]
    y_pred = mal_ll_score.tolist() + benign_ll_score.tolist()
    mean_fpr = np.linspace(0, 1, 1000)
    fpr,tpr,_ = roc_curve(y_true,y_pred)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return [roc_auc_score(y_true, y_pred),mean_fpr,interp_tpr]

def graphqa_rocv2(good_response_list,mal_response_list,fraction,ll_ranking_list,topk):
    #### select 200 training samples for likelihood score computation
    ll_score = []
    nqa = len(ll_ranking_list)
    ngood = good_response_list.shape[0]
    nmal = mal_response_list.shape[0]
    mal_data_idx = np.array(range(0,nmal))
    good_data_idx = np.array(range(0,ngood))
    np.random.shuffle(mal_data_idx)
    np.random.shuffle(good_data_idx)
    good_train_num = int(ngood * fraction)
    #good_test_num = ngood - good_train_num
    mal_train_num = int(nmal * fraction)
    #mal_test_num = nmal - mal_train_num
    good_train_idx = good_data_idx[:good_train_num]
    good_test_idx = good_data_idx[good_train_num:]
    mal_train_idx = mal_data_idx[:mal_train_num]
    mal_test_idx = mal_data_idx[mal_train_num:]
 
    for k in range(nqa):
        ll = (float(np.sum(mal_response_list[mal_train_idx,ll_ranking_list[k]])) / float(mal_train_num)) / (1e-5 + float(np.sum(good_response_list[good_train_idx,ll_ranking_list[k]])) / float(good_train_num))
        ll_score.append(np.log(ll+1e-5))
    
    ### testing 
    benign_ll_score = np.zeros(good_response_list[good_test_idx,:].shape[0])
    mal_ll_score = np.zeros(mal_response_list[mal_test_idx,:].shape[0])

    for k in range(len(good_test_idx)):
        for i in range(topk):
            if good_response_list[good_test_idx[k]][ll_ranking_list[i]] > 0:
                benign_ll_score[k] += ll_score[i]

    for k in range(len(mal_test_idx)):
        for i in range(topk):
            if mal_response_list[mal_test_idx[k]][ll_ranking_list[i]] > 0:
                mal_ll_score[k] += ll_score[i]

    y_true = len(mal_test_idx)*[1] + len(good_test_idx)*[0]
    y_pred = mal_ll_score.tolist() + benign_ll_score.tolist()
    mean_fpr = np.linspace(0, 1, 1000)
    fpr,tpr,_ = roc_curve(y_true,y_pred)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return [roc_auc_score(y_true, y_pred),mean_fpr,interp_tpr]

fraction = 0.9
nround = 50 
auc_score_round = []
mean_fprs = []
interp_tprs = []

for iround in range(nround):
    print(iround)
    auc_score_list = []
    for fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        auc_score, mean_fpr, interp_tpr = graphqa_roc(good_response_list_all,mal_response_list,fraction,30)
        if fraction == 0.4:
            mean_fprs.append(mean_fpr)
            interp_tprs.append(interp_tpr)

        auc_score_list.append(auc_score)
    
    auc_score_round.append(auc_score_list)


auc_score_round = np.array(auc_score_round)
print(auc_score_round.shape)
print(np.mean(auc_score_round,axis=0))
print(np.std(auc_score_round,axis=0))

#### apply for the 200 new benignware samples 
file_folder = '/benignware_200_graph_query_prompt_response.csv'
with open(file_folder,'r') as f:
    file_set = f.readlines()

file_hd = file_set[0].split('\t')
file_hash = []
good_response_list_test = []
for k in range(1,len(file_set)):
    records = file_set[k].split('\t')
    file_hash.append(records[0])
    good_response_list_test.append([int(records[j]) for j in range(1,len(records))])

question_list_good_test = [file_hd[j] for j in range(1,len(file_hd)-1)]
good_response_list_test = np.array(good_response_list_test)
nqa_test = len(question_list_good_test)
if nqa_test == nqa:
    topk = [30,60,90,100,120,150]
    n_test = good_response_list_test.shape[0]
    for val in topk:
        benign_ll_score_test = np.zeros(n_test)
        for k in range(n_test):
            for i in range(val): ### top feature number
                if good_response_list_test[k][ll_ranking_list[i]] > 0:
                    benign_ll_score_test[k] += ll_score[ll_ranking_list[i]]
        
        threshold = 14.60847303
        print('topk: ' + str(val) + '\t' + str(len(np.where(benign_ll_score_test > threshold)[0]))+'\t' + str(len(np.where(mal_ll_score < threshold)[0])) + '\n')
    
else:
    print('Exception')


print(np.array(question_list_mal)[ll_ranking_list[0:100]])

#### save the top 100 questions
with open('top100_questions.csv','w') as f:
    for k in range(100):
        f.write(question_list_mal[ll_ranking_list[k]] + '\n')

