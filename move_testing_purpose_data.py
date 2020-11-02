import os

cmd = 'mc cp csail/bkim/guiding-gtamp/for_testing_generators.zip ./ --recursive'
os.system(cmd)
cmd = 'unzip for_testing_generators.zip -d ./'
os.system(cmd)


