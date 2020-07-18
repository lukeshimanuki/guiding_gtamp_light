import os
import socket
import time

print "Inside the upload generator"

print "inside the loop for uploading files"
#time.sleep(600)

file_name = 'sampler_performances_{}.zip'.format(socket.gethostname())
cmd = 'zip -r -qq {} generators/sampler_performances'.format(file_name)
os.system(cmd)

cmd = 'mc rm csail/bkim/guiding-gtamp/{}'.format(file_name)
os.system(cmd)

cmd = 'mc cp {} csail/bkim/guiding-gtamp/ --recursive'.format(file_name)
os.system(cmd)
