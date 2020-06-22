import os
import sys
import numpy as np
import pickle



def download_from_s3():
    command = 'mc cp csail/bkim/guiding-gtamp/sampler_weights/ ./ --recursive'
    print command
    os.system(command)


def send_to_s3(domain):
    weight_dir = './generators/learning/learned_weights/{}/'.format(domain)
    algos = ['wgandi','wgangp','importance']
    seeds = range(4)
    atypes = ['place','pick']
    
    for algo in algos:
        for seed in seeds:
            for atype in atypes:  
                if atype == 'pick':
                    fdir  = './generators/learning/learned_weights/{}/{}/{}/fc/seed_{}/'.format(domain,atype,algo,seed)
                    command = 'mc cp {} {} --recursive'.format(fdir, 'csail/bkim/sampler_weights/')
                    os.system(command)
                else:
                    fdir  = './generators/learning/learned_weights/{}/{}/'.format(domain,atype)
                    regions = os.listdir(fdir)
                    for region in regions:
                        fdir  = './generators/learning/learned_weights/{}/{}/{}/{}/fc/seed_{}/'.format(domain,atype,region,algo,seed)
                        command = 'mc cp {} {} --recursive'.format(fdir, 'csail/bkim/sampler_weights/')
                        os.system(command)


def main():
    if sys.argv[1] == 'upload':
        send_to_s3(sys.argv[2])
    elif sys.argv[1] == 'download':
        download_from_s3()


if __name__ == '__main__':
    main()
