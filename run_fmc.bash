datasets=(adult)
norms=(L2)
lrs=(1e-2)

for dataset in ${datasets[@]}
do
    for norm in ${norms[@]}
    do
        for lr in ${lrs[@]}
        do
            python run_fmc.py --lmda_list 1 10 \
                                --sample_K 10 --data_set $dataset --n_sensitive 2 \
                                --max_iter 200 --m_step_iter 10 \
                                --cov_type 'isotropic' --use_cuda 'cuda:0' --lr $lr \
                                --tol_NLL 1e-6 \
                                --m_tole 1e-6 \
                                --random_state_list 1 \
                                --t_ratio 0.05 \
                                --sub_lmda 10 \
                                --is_normal $norm
        done
    done
done