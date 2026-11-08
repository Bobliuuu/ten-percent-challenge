set -e
cd ~/work
python3 -m venv ~/v2env && ~/v2env/bin/pip install -q -U pip uv
~/v2env/bin/uv pip install -p ~/v2env/bin/python vllm scipy hf_transfer safetensors huggingface_hub datasets
echo SETUP_DONE
