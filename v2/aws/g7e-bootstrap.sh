#!/bin/bash
# First-boot user data for 10%-challenge compute hosts (rendered by ~/inference-experiment/scripts/aws_g7e.py).
# Root part: deadline guard (poweroff -> terminate), SSH key, local NVMe. User part: python env for compression/feedback/probe jobs.
set -uo pipefail
exec >>/var/log/g7e-bootstrap.log 2>&1
echo "$(date -u +%FT%TZ) bootstrap start; deadline @DEADLINE_ISO@"
cat >/etc/systemd/system/g7e-deadline.service <<'EOT'
[Unit]
Description=Power off at the deadline (the instance then terminates)
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl poweroff
EOT
cat >/etc/systemd/system/g7e-deadline.timer <<'EOT'
[Unit]
Description=deadline @DEADLINE_ISO@
[Timer]
OnCalendar=@DEADLINE_CAL@
Persistent=true
AccuracySec=1s
[Install]
WantedBy=timers.target
EOT
systemctl daemon-reload && systemctl enable --now g7e-deadline.timer
install -d -m 700 -o ubuntu -g ubuntu /home/ubuntu/.ssh
echo '@PUBKEY@' >>/home/ubuntu/.ssh/authorized_keys
chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys && chmod 600 /home/ubuntu/.ssh/authorized_keys
N=/opt/dlami/nvme
if ! mountpoint -q "$N"; then
  dev=$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/ {print $1; exit}')
  if [[ -n "$dev" ]]; then mkfs.ext4 -q -F "$dev" && mkdir -p "$N" && mount -o noatime "$dev" "$N"; fi
fi
mountpoint -q "$N" || mkdir -p "$N"
chown ubuntu:ubuntu "$N"
nvidia-smi --query-gpu=name,memory.total --format=csv
cat >/home/ubuntu/bootstrap-user.sh <<'EOT'
#!/bin/bash
set -uo pipefail
N=/opt/dlami/nvme
mkdir -p $N/cache $N/work $N/hf
ln -sfn $N/work $HOME/work
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=$N/cache/uv HF_HOME=$N/hf TMPDIR=$N/cache
curl -LsSf https://astral.sh/uv/install.sh | sh
cd $N/work && uv venv --python 3.11 venv
. venv/bin/activate
uv pip install "vllm==0.11.0" "transformers==4.57.1" accelerate datasets scipy numpy safetensors constriction huggingface_hub[hf_xet] sentencepiece
git clone --depth 1 https://github.com/Bobliuuu/ten-percent-challenge.git repo || true
echo "export HF_HOME=$N/hf" >> $HOME/.bashrc
echo "$(date -u +%FT%TZ) ready" > $HOME/READY
EOT
chown ubuntu:ubuntu /home/ubuntu/bootstrap-user.sh && chmod +x /home/ubuntu/bootstrap-user.sh
sudo -u ubuntu nohup bash /home/ubuntu/bootstrap-user.sh >/home/ubuntu/bootstrap-user.log 2>&1 &
echo "$(date -u +%FT%TZ) root part done"
