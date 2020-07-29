import os

def main():
    cmd = 'cp ~/.kube/lis_config ~/.kube/config'
    os.system(cmd)
    algos = ['greedy-qlearned-hcount']
    algos = ['greedy-qlearned']
    abs_q_loss = 'largemargin'
    yaml_file = 'evaluate_task_guidance.yaml'
    num_trains = [5000]
    absqseeds = [0, 1, 2, 3]
    for num_train in num_trains:
        for algorithm in algos:
            for absq_seed in absqseeds:
                if algorithm == 'greedy-qlearned':
                    hoption = 'qlearned'
                else:
                    hoption = 'qlearned_hcount_old_number_in_goal'

                cmd = 'cat cloud_scripts/{} | ' \
                      'sed \"s/NAME/evaltaskguide-qseed-{}-loss-{}-hop-{}/\" | ' \
                      'sed \"s/HOPTION/{}/\" |  ' \
                      'sed \"s/ABSQSEED/{}/\" |  ' \
                      'sed \"s/ABSQLOSS/{}/\" |  ' \
                      'sed \"s/NUMTRAIN/{}/\" |  ' \
                      'kubectl apply -f - -n beomjoon;'.format(yaml_file, absq_seed, abs_q_loss, algorithm,
                                                               hoption,
                                                               absq_seed,
                                                               abs_q_loss,
                                                               num_train)
                print cmd
                os.system(cmd)

if __name__ == '__main__':
    main()
