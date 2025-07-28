cd ../src

../../parallel -j2 --resume-failed --results ../Output/weights_chimps_3 --joblog ../joblog/weights_chimps_3 CUDA_VISIBLE_DEVICES=3 python ./learn_weights.py --chimps --seed {1} --initial_values 0.1 0.1 0.1 1.0 0.1 0.1 0.1 0.1 0.1 --initial_addition {2} ::: 0 10 20 ::: 0.1

../../parallel -j2 --resume-failed --results ../Output/weights_syn_3 --joblog ../joblog/weights_syn_3 CUDA_VISIBLE_DEVICES=3 python ./learn_weights.py --seed {1} --uniform_initial_value 0.1 --initial_alternative_ind {2} --ph_list 0.1 0.1 0.1 0.1 0.1 --w_list 1 0 1 1 0 --w_add {3} ::: 0 10 20 ::: 0 1 2 3 4  ::: 0 0.3
