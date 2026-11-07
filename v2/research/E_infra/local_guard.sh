#!/bin/bash
# local_guard.sh REGION INSTANCE_ID SECONDS : backup terminate guard (Mac has no systemd)
sleep "$3"; AWS_PROFILE=lab aws ec2 terminate-instances --region "$1" --instance-ids "$2"
