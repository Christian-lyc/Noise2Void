#!/usr/bin/env bash

nohup python3 -u main.py >log.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
