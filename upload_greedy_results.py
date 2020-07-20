import os
import socket
import time

print "Inside the upload generator"

while True:
    print "inside the loop for uploading files"

    file_name = 'test_results_{}.zip'.format(socket.gethostname())
    cmd = 'zip -r -qq {} ./test_results'.format(file_name)
    os.system(cmd)

    cmd = 'mc rm csail/bkim/guiding-gtamp/{}'.format(file_name)
    os.system(cmd)

    cmd = 'mc cp {} csail/bkim/guiding-gtamp/ --recursive'.format(file_name)
    os.system(cmd)
    time.sleep(300)
