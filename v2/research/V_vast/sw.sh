#!/bin/bash
exec ssh -i ~/.ssh/swarm-gpu -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 ubuntu@13.223.68.70 "$@"
