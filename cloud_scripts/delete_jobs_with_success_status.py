import subprocess
import time
import os

while True:
    cmd = 'kubectl get job -o=jsonpath=\'{.items[?(@.status.succeeded==1)].metadata.name}\' -n beomjoon'
    result = subprocess.check_output(cmd, shell=True)
    for done_job in result.split(' '):
        cmd = 'kubectl delete job ' + done_job + ' -n beomjoon'
        time.sleep(0.5)
        os.system(cmd)
    print "Sleeping..."
    time.sleep(120)
