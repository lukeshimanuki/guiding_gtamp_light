#!/bin/bash

cd /guiding_gtamp_light
sed -i "s/url = git@github.com:/url = http:\/\/github.com\//" .git/config
git config --global user.email "beomjoon.kim0@gmail.com"
git fetch --all
git reset --hard origin/beomjoon
git checkout -- .
git log --pretty=format:%h -n 1
mc config host add csail https://ceph.csail.mit.edu 119df3538de4475aaf3f0737600003b4 7e0b2e1e15004eafb75f49d1f265baf7
mc cp csail/bkim/ik/kinematics.9ff4f1d77a61494bbd09f843fedb0314 ~/.openrave/ --recursive
mc cp csail/bkim/ik/kinematics.4f95c55204252b6edd6332624a20624c ~/.openrave/ --recursive
mc cp csail/bkim/ik/robot.7edbf73fb4fc856e8294d93279d26ff2 ~/.openrave/ --recursive
export PYTHONPATH=/guiding_gtamp_light/openrave_wrapper/manipulation:/guiding_gtamp_light/openrave_wrapper:/guiding_gtamp_light:/guiding_gtamp_light/mover_library:/pddlstream