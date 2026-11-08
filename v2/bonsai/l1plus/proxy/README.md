# Hostile OOD proxy set for the voice-skill router

Hand-written by an LLM agent (text authored directly, utterances invented); no model scored or was run on it. Nothing from v2/exam or sealed files was read.

Files (JSONL rows: id, text, kind, style):
- old_skills.jsonl: 315 rows, 45 per skill (Alarm, Calendar, Lists, Music, News, Transport, Weather)
- new_translate.jsonl: 130 rows, kind=Translate
- new_banking.jsonl: 130 rows, kind=Banking
- new_email.jsonl: 130 rows, kind=Email

Style tags (roughly equal shares per file):
a = British/Indian/Singlish/Aus phrasing; b = ASR-noise (lowercase, no punctuation, dropped words, homophones);
c = typos/abbreviations/texting slang; d = long polite multi-clause LLM-style (25-60 words) with context;
e = conversational/indirect; f = mixed-language lite (Hinglish/Spanglish); g = terse keywords.
Old-skill labels were chosen to be unambiguous to a human; some (b/c/g) rows are intentionally hard for a model.
