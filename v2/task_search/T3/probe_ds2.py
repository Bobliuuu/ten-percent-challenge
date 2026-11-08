import sys, signal
from datasets import load_dataset
C = [
("heegyu/news-category-dataset",None,"train"),("ucberkeley-dlab/measuring-hate-speech",None,"train"),("raquiba/Sarcasm_News_Headline",None,"train"),
("zeroshot/twitter-financial-news-sentiment",None,"train"),("gtfintechlab/fomc_communication",None,"train"),("ccdv/arxiv-classification",None,"train"),
("Jean-Baptiste/financial_news_sentiment",None,"train"),("mteb/toxic_conversations_50k",None,"train"),("mteb/amazon_massive_scenario","en","train"),
("cybersectony/PhishingEmailDetectionv2.0",None,"train"),("puyang2025/seven-phishing-email-datasets",None,"train"),("kmack/Phishing_urls",None,"train"),("drorrabin/phishing_emails-data",None,"train"),
("facebook/anli",None,"train_r1"),("tasksource/sarcasm",None,"train"),("nampdn-ai/tiny-codes",None,"train"),("Muennighoff/humanevalpack","python","test"),("claudios/code_search_net","python","train"),
("google/code_x_glue_cc_defect_detection",None,"train"),("DetectVul/devign",None,"train"),("allenai/csabstruct",None,"train"),
("PKU-Alignment/PKU-SafeRLHF",None,"train"),("OpenSafetyLab/Salad-Data","base_set","train"),("yahma/alpaca-cleaned",None,"train"),("databricks/databricks-dolly-15k",None,"train"),
("textdetox/multilingual_toxicity_dataset","en","train"),("tasksource/jigsaw_toxicity",None,"train"),("google/jigsaw_toxicity_pred",None,"train"),
("SetFit/enron_spam",None,"train"),("ucirvine/sms_spam",None,"train"),("rajpurkar/squad_v2",None,"train"),("microsoft/ms_marco","v2.1","train"),
("clinc/clinc_oos","plus","train"),("tasksource/clickbait",None,"train"),("marksverdhei/clickbait_title_classification",None,"train"),
("amazon_reviews: McAuley-Lab/Amazon-Reviews-2023",None,"train"),("SetFit/amazon_counterfactual_en",None,"train"),("SetFit/amazon_reviews_multi_en",None,"train"),
("mteb/amazon_massive_domain","en","train"),("yelp_review_full",None,"train"),("Yelp/yelp_review_full",None,"train"),
("Johnnyeee/Ecommerce-product-category",None,"train"),("kaggle resumes: ahmedheakl/resume-atlas",None,"train"),("ahmedheakl/resume-atlas",None,"train"),
("victor/real-or-fake-fake-jobposting-prediction",None,"train"),("Nahid1992/fake_job_posting",None,"train"),
("liar: ucsbnlp/liar",None,"train"),("GonzaloA/fake_news",None,"train"),("mteb/hotel-reviews",None,"train"),
("deepset/prompt-injections",None,"train"),("jackhhao/jailbreak-classification",None,"train"),
("ccdv/patent-classification",None,"train"),("legacy-datasets/wikipedia",None,"train"),
("nlpaueb/multi_eurlex",None,"train"),("lex_glue: coastalcph/lex_glue","case_hold","train"),("LawInformedAI/claudette_tos",None,"train"),
("fancyzhx/dbpedia_14",None,"train"),("community-datasets/yahoo_answers_topics",None,"train"),("SetFit/bbc-news",None,"train"),
("stanfordnlp/sentiment140",None,"train"),("mteb/emotion",None,"train"),("ought/raft","banking_77","train"),("ought/raft","terms_of_service","train"),("ought/raft","tai_safety_research","train"),("ought/raft","ade_corpus_v2","train"),("ought/raft","systematic_review_inclusion","train"),("ought/raft","overruling","train"),("ought/raft","twitter_complaints","train"),("ought/raft","neurips_impact_statement_risks","train"),
("ade-benchmark-corpus/ade_corpus_v2","Ade_corpus_v2_classification","train"),("bigbio/bc5cdr",None,"train"),
("openai/gsm8k","main","train"),("TIGER-Lab/MMLU-Pro",None,"test"),
]
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM,h)
for n,cfg,sp in C:
    if ":" in n: n=n.split(": ")[-1]
    signal.alarm(40)
    try:
        d=load_dataset(n,cfg if cfg and cfg not in("x",) else None,split=sp,streaming=True) if True else None
        r=next(iter(d)); print("OK ",n,cfg,list(r.keys())[:8],str(r)[:120].replace("\n"," "),flush=True)
    except TO: print("TIMEOUT",n,flush=True)
    except Exception as e: print("FAIL",n,cfg,type(e).__name__,str(e)[:90].replace("\n"," "),flush=True)
    signal.alarm(0)
