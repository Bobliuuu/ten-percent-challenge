"""Copied verbatim from v2/lanes/L9b/final_jobs.py (same prompts/sets as the L9b own-data dry run)."""
import random, urllib.request, csv, io
TIMER_T = ["set a timer for {d}", "start a {d} timer", "timer for {d} please", "can you count down {d} for me", "i need a {d} countdown timer", "begin a timer, {d}", "ping me when {d} are up", "how about a {d} timer"]
DUR = ["ten minutes", "5 minutes", "half an hour", "45 seconds", "two hours", "twenty minutes", "90 seconds", "an hour"]
def timer_sets():
    rows = [t.format(d=d) for t in TIMER_T for d in DUR]; random.Random(7).shuffle(rows); return rows[:20], rows[20:48]

BK_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
def banking_rows(n=200):
    txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); rs = list(csv.DictReader(io.StringIO(txt))); random.Random(11).shuffle(rs)
    return [(r["text"], r["category"]) for r in rs[:n]]
def generic_prompts():
    P = []
    for a, b in [(17, 23), (48, 12), (305, 7), (91, 14), (56, 56), (123, 4), (77, 33), (250, 18), (64, 9), (19, 21)]: P.append(f"What is {a} times {b}? Answer with just the number.")
    for c in ["France", "Japan", "Brazil", "Kenya", "Canada", "Egypt", "Norway", "Peru", "India", "Australia"]: P.append(f"What is the capital of {c}? One word.")
    for w in ["happy", "large", "quickly", "ancient", "silent", "bright", "fragile", "generous", "rapid", "narrow"]: P.append(f"Give one antonym of '{w}'. One word.")
    for w in ["good morning", "thank you", "where is the station", "i would like some water", "how much does this cost", "see you tomorrow", "i am tired", "the book is on the table", "what time is it", "nice to meet you"]: P.append(f"Translate to French: {w}")
    for t in ["photosynthesis", "inflation", "a hash table", "gravity", "the TCP handshake", "a black hole", "compound interest", "DNA replication", "a binary search", "the water cycle"]: P.append(f"Explain {t} in one short sentence.")
    for t in ["reverse a string in Python", "sum a list in Python", "check if a number is prime in Python", "read a file in Python", "sort a dictionary by value in Python", "make an HTTP GET request in Python", "print the numbers 1 to 10 in Python", "count words in a string in Python", "flatten a nested list in Python", "get the current date in Python"]: P.append(f"Write a one-line Python snippet to {t}.")
    for s in ["The quick brown fox", "Once upon a time", "In the beginning", "It was a dark and stormy night", "The meeting is scheduled for", "Dear customer,", "To install the package, run", "The main advantage of", "Breaking news:", "Step one:"]: P.append(f"Continue this sentence in about ten words: {s}")
    for q in ["Is a tomato a fruit or a vegetable?", "Which is heavier, a kilogram of feathers or a kilogram of steel?", "Is 91 prime?", "How many days are in a leap year?", "Name the largest planet in the solar system.", "Who wrote Pride and Prejudice?", "What is the chemical symbol for gold?", "How many sides does a hexagon have?", "What language is spoken in Brazil?", "What is the boiling point of water in Celsius?"]: P.append(q)
    for t in ["a cat", "the ocean", "autumn", "coffee", "a train", "winter", "a library", "the moon", "friendship", "a city at night"]: P.append(f"Write a haiku about {t}.")
    for t in ["my landlord about a broken heater", "a colleague to thank them for their help", "a friend to invite them to dinner", "a hotel to ask about late checkout", "a teacher to ask for an extension", "customer support about a late delivery", "my manager to request a day off", "a neighbour about loud music", "a bank about a lost card", "a recruiter to decline an offer"]: P.append(f"Write a two-sentence email to {t}.")
    for t in ["bake bread", "tie a tie", "change a tire", "boil an egg", "plant a tree", "write a cover letter", "learn to swim", "make tea", "clean a keyboard", "start a vegetable garden"]: P.append(f"Give me three short tips to {t}.")
    return P

