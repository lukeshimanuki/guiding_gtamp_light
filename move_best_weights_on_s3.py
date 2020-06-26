import os
import sys


def download_from_s3():

    command = 'mc cp csail/bkim/guiding-gtamp/sampler_weights/learned_weights.zip ./ --recursive'
    print command
    os.system(command)
    command = 'unzip learned_weights.zip -d ./'
    print command
    os.system(command)


def send_to_s3(domain):
    cmd = 'zip -r learned_weights.zip generators/learning/learned_weights'
    os.system(cmd)
    cmd = 'mc cp learned_weights.zip csail/bkim/guiding-gtamp/sampler_weights/ --recursive'
    os.system(cmd)
    return


def main():
    if sys.argv[1] == 'upload':
        send_to_s3(sys.argv[2])
    elif sys.argv[1] == 'download':
        download_from_s3()


if __name__ == '__main__':
    main()
