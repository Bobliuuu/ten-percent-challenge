# Hostile OOD proxy set for the bank support-chat intent router (L3)

Hand-written by an LLM agent (text authored directly, utterances invented); no model scored or was run on them. Nothing sealed (nothing from v2/exam, no lock or exam files) was read. Format and style tags mirror v2/bonsai/l1plus/proxy/.

Files (JSONL rows: id, text, kind, style):
- bank_old.jsonl: 180 rows, 30 per team (Card, Transfer, TopUp, Cash, Identity, Exchange), ids px-bo-0001..0180
- new_security.jsonl: 130 rows, kind=Security (fraud, phishing, account takeover, scams, unauthorised payments, compromised details); a NEW team not in the router menu, ids px-bsec-0001..0130

Task prompt: "You are the intent router of a bank's support chat. Decide which team the customer's message belongs to. Answer with exactly one word from: Card, Transfer, TopUp, Cash, Identity, Exchange."

Style tags (roughly equal shares per file):
a = British/Indian/Singlish/Aus phrasing; b = ASR-noise (lowercase, no punctuation, dropped words, homophones);
c = typos/abbreviations/texting slang; d = long polite multi-clause LLM-style (25-60 words) with context;
e = conversational/indirect; f = mixed-language lite (Hinglish/Spanglish); g = terse keywords.
Old-team labels were chosen to be unambiguous to a human; some (b/c/g) rows are intentionally hard for a model.
