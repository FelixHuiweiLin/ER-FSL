import time
import numpy as np
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from experiment.metrics import compute_performance, single_run_avg_end_fgt
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
import os
import pickle

def multiple_run(params, store=False, save_path=None):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if store:
        result_path = load_yaml('config/global.yml', key='path')['result']
        table_path = result_path + params.data
        print(table_path)
        os.makedirs(table_path, exist_ok=True)
        if not save_path:
            save_path = params.model_name + '_' + params.data_name + '.pkl'

    accuracy_list = []
    for run in range(params.num_runs):
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        model = setup_architecture(params)
        model = maybe_cuda(model, params.cuda)
        opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
        agent = agents[params.agent](model, opt, params)

        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            print("-----------run {} training batch {}-------------".format(run, i))
            print('size: {}, {}'.format(x_train.shape, y_train.shape))
            agent.train_learner(x_train, y_train)
            acc_array = agent.evaluate(test_loaders)
            tmp_acc.append(acc_array)
        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

    accuracy_array = np.array(accuracy_list)
    end = time.time()
    if store:
        result = {'time': end - start}
        result['acc_array'] = accuracy_array
        save_file = open(table_path + '/' + save_path, "wb")
        pickle.dump(result, save_file)
        save_file.close()
    if params.online:
        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    else:
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print("avg_end_acc {}".format(np.mean(accuracy_list)))

