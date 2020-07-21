import os


def copy_contents(source_f, target_f):
    source_lines = open(source_f, 'r').readlines()
    target_lines = open(target_f, 'r').readlines()

    target_write = open(target_f, 'a')
    for s in source_lines:
        if s not in target_lines:
            target_write.write(s)


def main():
    smpler_result_path = 'generators/sampler_performances/'
    atype = 'place_loading'
    results_by_machines = [f for f in os.listdir(smpler_result_path)]

    algo = 'wgandi'

    save_path_root = 'generators/sampler_performances/'
    for machine_name in results_by_machines:
        if machine_name == atype or machine_name == 'with_goal_object_poses' or machine_name == 'pick':
            continue
        machine_path = smpler_result_path + machine_name + '/' + atype
        save_path = save_path_root + '/' + atype
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        try:
            seed_dirs = os.listdir(machine_path)
        except:
            import pdb;pdb.set_trace()
        for seed_dir in seed_dirs:
            if os.path.isdir(save_path + '/' + seed_dir):
                save_result_dir = save_path + '/' + seed_dir + '/' + algo + '/'
                machine_result_dir = machine_path + '/' + seed_dir + '/' + algo + '/'

                save_epoch_files = os.listdir(save_result_dir)
                machine_epoch_files = os.listdir(machine_result_dir)
                for m_epoch_f in machine_epoch_files:
                    if m_epoch_f in save_epoch_files:
                        copy_contents(machine_result_dir + m_epoch_f, save_result_dir + m_epoch_f)
                    else:
                        cmd = 'cp -r {} {}'.format(machine_result_dir + '/' + m_epoch_f, save_result_dir)
                        print cmd
                        os.system(cmd)

            else:
                cmd = 'cp -r {} {}'.format(machine_path + '/' + seed_dir, save_path + '/')
                print cmd
                os.system(cmd)


if __name__ == '__main__':
    main()
