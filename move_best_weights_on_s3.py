import os
import sys
import socket


def download_from_s3():
    command = 'mc cp csail/bkim/guiding-gtamp/sampler_weights/learned_weights_phaedra.zip ./ --recursive'
    print command
    os.system(command)

    command = 'unzip learned_weights_phaedra.zip -d ./'
    print command
    os.system(command)

    command = 'mc cp csail/bkim/guiding-gtamp/sampler_weights/learned_weights_shakey.zip ./ --recursive'
    print command
    os.system(command)

    command = 'unzip learned_weights_shakey.zip -d ./'
    print command
    os.system(command)


def send_to_s3(domain):
    file_name = 'learned_weights_{}.zip'.format(socket.gethostname())

    cmd = 'mc rm ./{}'.format(file_name)
    print cmd
    os.system(cmd)

    cmd = 'mc rm csail/bkim/guiding-gtamp/sampler_weights/{}'.format(file_name)
    print cmd
    os.system(cmd)

    cmd = 'zip -r {} generators/learning/learned_weights'.format(file_name)
    print cmd
    os.system(cmd)

    cmd = 'mc cp {} csail/bkim/guiding-gtamp/sampler_weights/ --recursive'.format(file_name)
    print cmd
    os.system(cmd)
    return


def main():
    if sys.argv[1] == 'upload':
        send_to_s3(sys.argv[2])
    elif sys.argv[1] == 'download':
        download_from_s3()


if __name__ == '__main__':
    main()
