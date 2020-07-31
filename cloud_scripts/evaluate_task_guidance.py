import os


def main():
    cmd = 'cp ~/.kube/lis_config ~/.kube/config'
    os.system(cmd)
    abs_q_losses = ['largemargin', 'mse']
    algos = ['greedy-qlearned-hcount'] #, 'greedy-qlearned']
    yaml_file = 'evaluate_task_guidance.yaml'
    num_trains = [100, 1000, 2000, 3000, 4000]
    absqseeds = [0, 1, 2, 3]
    for abs_q_loss in abs_q_losses:
        for planseed in [0, 1, 2, 3]:
            for num_train in num_trains:
                for algorithm in algos:
                    for absq_seed in absqseeds:
                        if algorithm == 'greedy-qlearned':
                            hoption = 'qlearned'
                        else:
                            hoption = 'qlearned_hcount_old_number_in_goal'

                        cmd = 'cat cloud_scripts/{} | ' \
                              'sed \"s/NAME/taskguide-{}-{}-{}-{}-{}/\" | ' \
                              'sed \"s/HOPTION/{}/\" |  ' \
                              'sed \"s/ABSQSEED/{}/\" |  ' \
                              'sed \"s/ABSQLOSS/{}/\" |  ' \
                              'sed \"s/NUMTRAIN/{}/\" |  ' \
                              'sed \"s/PLANSEED/{}/\" |  ' \
                              'kubectl apply -f - -n beomjoon;'.format(yaml_file, num_train, absq_seed, abs_q_loss, planseed, algorithm,
                                                                       hoption,
                                                                       absq_seed,
                                                                       abs_q_loss,
                                                                       num_train,
                                                                       planseed)
                        print cmd
                        os.system(cmd)


if __name__ == '__main__':
    main()
