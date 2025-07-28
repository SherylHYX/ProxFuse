cd ../src

../../parallel -j2 --resume-failed --results ../Output/weights_chimps_0 --joblog ../joblog/weights_chimps_0 CUDA_VISIBLE_DEVICES=0 python ./learn_weights.py --chimps --seed {1} --initial_values 1.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 --initial_addition {2} ::: 0 10 20  ::: 0.1

../../parallel -j2 --resume-failed --results ../Output/weights_syn_0 --joblog ../joblog/weights_syn_0 CUDA_VISIBLE_DEVICES=0 python ./learn_weights.py --seed {1} --uniform_initial_value {2} --ph_list 0.1 0.1 0.1 0.1 0.1 --w_list 1 0 1 0 1 --w_add {3} ::: 0 10 20 ::: 0.1 0.2 0.5 1.0 2.0 5.0 ::: 0 0.3
