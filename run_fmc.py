import argparse
import torch
import numpy as np
from utils.utils_mode import to_numpy, resp_, sorted_counter, calculate_Fscore, re_calc_distance
from utils.utils_data import read_dataset
from FMC_EM import Gaussian_mixture as em_fair
import time
import os
import pandas as pd
from torch.utils.data import random_split

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

def main(args):
    ####### parser ######
    lmdas = (args.lmda_list)
    data_set = args.data_set
    max_iter = args.max_iter
    m_step_iter = args.m_step_iter
    n_sensitive = args.n_sensitive
    lr = args.lr
    cov_type = args.cov_type
    tol_NLL = args.tol_NLL
    m_tole = args.m_tole
    ipynb_cuda = args.use_cuda
    random_state_list = args.random_state_list
    sample_K = args.sample_K
    t_ratio = args.t_ratio
    sub_lmda = args.sub_lmda
    is_normal = args.is_normal

    data_dir = './datasets/'
    if data_set == 'adult':
        cont_data, cate_data, sens_data, clss_data = read_dataset(data_dir, data_set, n_sensitive, normalize = is_normal)
    else:
        cont_data, cate_data, sens_data = read_dataset(data_dir, data_set, n_sensitive, normalize = is_normal)

    if data_set == 'census':
        n_train = int(cont_data.shape[0] * t_ratio)
        n_test = cont_data.shape[0] - n_train
        train_data, test_data = random_split(sens_data, [n_train, n_test])
        train_index = train_data.indices
        test_index = test_data.indices
        cont_test = cont_data[test_index]
        cate_test = None
        sens_test = sens_data[test_index]
        cont_data = cont_data[train_index]
        cate_data = None
        sens_data = sens_data[train_index]
        
    print('============================================================')
    print(f'Data_set: {data_set}')
    print(f'Covariance: {cov_type}')
    print(f'Use_cuda: {ipynb_cuda}')
    print(f'Random State List: {random_state_list}')
    print(f'Lambda: {lmdas}')
    print(f'Sub Lambda: {sub_lmda}')
    print(f'Learning rate: {lr}')
    print(f'NLL tolerance: {tol_NLL}')
    print(f'M-step tolerance: {m_tole}')
    print(f'Max Iteration: {max_iter}')
    print(f'Max M-step Iteration: {m_step_iter}')
    print(f'Train Ratio: {t_ratio}')
    print(F'Normalization: {is_normal}')
    print('============================================================')
    
    for random_state in random_state_list:
        print('============================================================')
        print(f'Seed: {random_state}')
        print('------------------------------------------------------------')
        for lmda in lmdas:
            st = time.time()
            fair_em = em_fair(n_components=sample_K,
                            data_cont = cont_data,
                            data_cate = cate_data,
                            sensitive = sens_data, lmda = lmda,
                            cov_type=cov_type, random_state=random_state, tol=tol_NLL,
                            max_iter=max_iter, device = ipynb_cuda,
                            lr = lr,
                            m_tole=m_tole,
                            m_step_iter = m_step_iter,
                            window_length=10,
                            verbose = False,
                            train_cov = True,
                            start_em = True,
                            sub_lmda = sub_lmda)
                        
            print(f'Covariance type: {fair_em.cov_type}')
            print(f'Lambda: {fair_em.lmda}')
            print(f'NLL tolerance: {fair_em.tol}')
            print(f'Learnig Rate: {fair_em.lr}')
            print(f'Use_cuda: {fair_em.device}')
            print(f'K: {fair_em.K}')
            fair_em.fit()
            print('============================================================')
            print(f'< < Result > >')
            print(f'Model Lambda: {fair_em.lmda}')
            print(f'NLL: {fair_em.NLL_list[-1]: 0.5f}')
            print(f'Balance: {fair_em.balance_list[-1]: 0.5f}')
            ed = time.time()
            total_took = ed - st
            hours = total_took // 3600
            mins = (total_took - hours*3600)//60
            seconds = total_took % 60
            print(f'Time: {total_took//3600} hours {mins} mins {seconds: 0.1f} seconds')
            print('============================================================')

            total_nll = fair_em.NLL_list[-1]
            total_fscore = fair_em.Fscore_list[-1]
            total_balance = fair_em.balance_list[-1]
            if fair_em.conts:
                total_dist = fair_em.dist_list[-1]
            else:
                total_dist = None

            total_result = {'Seed': [random_state],
                      'K': [sample_K],
                      'Set' : ['train'],
                      'Lambda': [lmda],
                      'Cost': [total_dist], 'NLL': [total_nll], 'Delta': [total_fscore], 'Balance': [total_balance],
                      'Inference time': [total_took]}
            total_df = pd.DataFrame(total_result)

            if data_set == 'census':
                st = time.time()
                np_mu = to_numpy(fair_em.means_)
                np_prec = to_numpy(fair_em.precisions_)
                np_weight = to_numpy(fair_em.weights_)
                test_score = resp_(cont_test, sample_K, np_mu, np_prec, np_weight)

                if (test_score.sum(axis = 1) == 0).any():
                    test_score[np.where(test_score.sum(axis = 1) == 0)[0][0]] = test_score[np.where(test_score.sum(axis = 1) == 0)[0][0]] + 1e-20
                test_nll = -np.log(test_score.sum(axis = 1)).mean()

                test_resp = test_score / test_score.sum(axis = 1, keepdims = True)
                
                test_predict = test_resp.argmax(axis = 1)
                test_predict_0 = sorted_counter(test_predict[sens_test == 0], sample_K)[:,1]
                test_predict_1 = sorted_counter(test_predict[sens_test == 1], sample_K)[:,1]
                test_balance = np.min([(test_predict_0 / test_predict_1).min(), (test_predict_1 / test_predict_0).min()])
                
                test_fscore = calculate_Fscore(test_resp, sens_test)
                _, test_euc = re_calc_distance(cont_test, test_predict, sample_K)
                ed = time.time()
                test_took = ed - st
                test_result = {'Seed': [random_state],
                        'K': [sample_K],
                        'Set' : ['test'],
                        'Lambda': [lmda],
                        'Cost': [test_euc], 'NLL': [test_nll], 'Delta': [test_fscore], 'Balance': [test_balance],
                        'Inference time': [test_took]}
                test_df = pd.DataFrame(test_result)
                total_df = pd.concat([total_df, test_df], ignore_index=True)

            path = f'results/{args.data_set}_{n_sensitive}/{is_normal}/K_{sample_K}/'
            os.makedirs(path, exist_ok = True)
            file_path = path + f'results.csv'
            if os.path.exists(file_path):
                total_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                total_df.to_csv(file_path, mode='w', header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMC')
    parser.add_argument('--lmda_list', nargs='+', type=float, help="List of numbers", default=[1, 10])
    parser.add_argument('--data_set', type=str, choices=['adult', 'bank', 'census', 'credit'], default='adult')
    parser.add_argument('--n_sensitive', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--m_step_iter', type=int, default=10)
    parser.add_argument('--cov_type', default='isotropic', choices=['diag', 'isotropic'])
    parser.add_argument('--use_cuda', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--tol_NLL', type=float, default=1e-6)
    parser.add_argument('--m_tole', type=float, default=1e-6)
    parser.add_argument('--random_state_list', nargs='+', type=int, default=[1])
    parser.add_argument('--sample_K', default=10, type=int)
    parser.add_argument('--t_ratio', default=0.01, type=float)
    parser.add_argument('--sub_lmda', default=10, type=float)
    parser.add_argument('--is_normal', choices=['L2', 'standard'])
    main(parser.parse_args())