import os
import sys
from test_scripts.run_generator import parse_arguments

cmd = 'python upload_generator_results.py &'
os.system(cmd)

generator_commands = ''
for arg in sys.argv[1:]:
    generator_commands += ' ' + arg

generator_commands = 'python test_scripts/multithreaded/run_generator.py ' + generator_commands
os.system(generator_commands)
