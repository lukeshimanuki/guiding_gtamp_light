import os
import socket
import time

print "Inside the upload generator"

file_name = 'test_results_{}.zip'.format(socket.gethostname())
os.system('rm {}'.format(file_name))
print 'Zipping....'
cmd = 'zip -r -qq {} ./test_results'.format(file_name)
print 'Done!'
os.system(cmd)

cmd = 'mc rm csail/bkim/guiding-gtamp/{}'.format(file_name)
print cmd
os.system(cmd)

cmd = 'mc cp {} csail/bkim/guiding-gtamp/ --recursive'.format(file_name)
print cmd
os.system(cmd)
