import os
import subprocess
import time
s3_path = 'csail/bkim/guiding-gtamp'
result = subprocess.check_output('mc ls {}'.format(s3_path), shell=True)

target_files = []
download_cmd = 'mc cp csail/bkim/guiding-gtamp/'
for l in result.split('\n'):
    filename = l.split(' ')[-1]
    if 'sampler_performances' in filename:
        download_cmd = 'mc cp csail/bkim/guiding-gtamp/{} ./'.format(filename)
        print download_cmd
        os.system(download_cmd)
        time.sleep(0.3)
        unzip_cmd = 'unzip -o -qq {} -d ./'.format(filename)
        os.system(unzip_cmd)


