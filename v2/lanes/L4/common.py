"""shared prompt templates (identical to the locked exam prompts) for answer.py / feedback.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = {"SKILL7": "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: Alarm, Weather, Music, Calendar, Lists, News, Transport.\n\nRequest: {X}",
     "BANK6": "You are the intent router of a bank's support chat. Decide which team the customer's message belongs to.\nAnswer with exactly one word from: Card, Transfer, TopUp, Cash, Identity, Exchange.\n\nMessage: {X}"}
def prompt_ids(tok, task, text):
    return tok(tok.apply_chat_template([{"role": "user", "content": P[task].replace("{X}", text)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
def cal(task): return [json.loads(l) for l in open(f"{HERE}/cal_{task.lower()}.jsonl")]
