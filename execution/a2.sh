cd ../src

../../parallel -j2 --resume-failed --results ../Output/simple_weights_chimps_2 --joblog ../joblog/simple_weights_chimps_2 CUDA_VISIBLE_DEVICES=2 python ./simple_learn_weights.py --chimps --seed {1} --initial_values 0.1 0.1 1.0 0.1 0.1 0.1 0.1 0.1 0.1 --initial_addition {2} ::: 0 10 20 ::: 0.1

../../parallel -j2 --resume-failed --results ../Output/simple_weights_syn_2 --joblog ../joblog/simple_weights_syn_2 CUDA_VISIBLE_DEVICES=2 python ./simple_learn_weights.py --seed {1} --uniform_initial_value {2} --ph_list 0.1 0.1 0.1 0.1 0.1 --w_list 1 0 1 1 0 --w_add {3} ::: 0 10 20 ::: 0.1 0.2 0.5 1.0 2.0 5.0 ::: 0 0.3
