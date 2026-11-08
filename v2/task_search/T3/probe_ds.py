import sys, signal
from datasets import load_dataset
C = [
("sms_spam",None,"train"),("Deysi/spam-detection-dataset",None,"train"),("SetFit/enron_spam",None,"train"),("ealvaradob/phishing-dataset","texts","train"),
("zefang-liu/phishing-email-dataset",None,"train"),
("stanfordnlp/sst2",None,"train"),("stanfordnlp/imdb",None,"train"),("fancyzhx/yelp_polarity",None,"train"),("fancyzhx/amazon_polarity",None,"train"),
("google/civil_comments",None,"train"),("Hello-SimpleAI/HC3",None,"train"),("tdavidson/hate_speech_offensive",None,"train"),("ethos","binary","train"),("Hate-speech-CNERG/hatexplain",None,"train"),("skg/toxigen-data","train","train"),
("nyu-mll/glue","rte","train"),("nyu-mll/glue","mrpc","train"),("nyu-mll/glue","qqp","train"),("stanfordnlp/snli",None,"train"),("nyu-mll/multi_nli",None,"train"),("allenai/scitail","tsv_format","train"),("google/boolq",None,"train"),
("lukasellinger/ag_news",None,"train"),("fancyzhx/ag_news",None,"train"),("SetFit/20_newsgroups",None,"train"),
("codeparrot/github-code-clean",None,"train"),("bigcode/the-stack-smol","data/python","train"),
("PubMed RCT: armanc/pubmed-rct20k",None,"train"),("ccdv/pubmed-summarization",None,"train"),
("coastalcph/lex_glue","unfair_tos","train"),("coastalcph/lex_glue","ledgar","train"),("nguha/legalbench","contract_nli_explicit_identification","train"),
("PolyAI/banking77",None,"train"),("clinc/clinc_oos","plus","train"),("tner/ner","x","train"),
("bitext/Bitext-customer-support-llm-chatbot-training-dataset",None,"train"),("Tobi-Bueck/customer-support-tickets",None,"train"),
("mteb/amazon_reviews_multi","en","train"),("mteb/tweet_sentiment_extraction",None,"train"),("cardiffnlp/tweet_eval","sentiment","train"),("cardiffnlp/tweet_eval","irony","train"),("cardiffnlp/tweet_eval","stance_abortion","train"),("cardiffnlp/tweet_eval","offensive","train"),("cardiffnlp/tweet_eval","hate","train"),
("dair-ai/emotion",None,"train"),("papluca/language-identification",None,"train"),("clips/mqa",None,"train"),
("lmsys/toxic-chat","toxicchat0124","train"),("Anthropic/hh-rlhf",None,"train"),("PKU-Alignment/BeaverTails",None,"330k_train"),("nvidia/Aegis-AI-Content-Safety-Dataset-2.0",None,"train"),("walledai/AdvBench",None,"train"),("allenai/wildguardmix","wildguardtrain","train"),("allenai/xstest-response",None,"response_harmfulness"),
("Salesforce/fineweb_deduplicated",None,"train"),("rajpurkar/squad_v2",None,"train"),("microsoft/ms_marco","v2.1","train"),
("heegyu/news-category-dataset",None,"train"),("ucberkeley-dlab/measuring-hate-speech",None,"train"),
("mteb/sarcasm_headlines",None,"train"),("raquiba/Sarcasm_News_Headline",None,"train"),("clickbait: christinacdl/clickbait_notclickbait_dataset",None,"train"),
("financial_phrasebank","sentences_allagree","train"),("zeroshot/twitter-financial-news-sentiment",None,"train"),("gtfintechlab/fomc_communication",None,"train"),
("nlpaueb/finer-139",None,"train"),("flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",None,"train"),
("emrecan/stsb-mt-turkish",None,"train"),("McAuley-Lab/Amazon-Reviews-2023","raw_review_Software","full"),
("lukesjordan/worldbank-project-documents",None,"train"),("mteb/mtop_domain","en","train"),("mteb/amazon_massive_scenario","en","train"),
("ccdv/arxiv-classification",None,"train"),("Jean-Baptiste/financial_news_sentiment",None,"train"),("mteb/toxic_conversations_50k",None,"train"),("mteb/reddit-clustering",None,"train"),
("jhu-clsp/jfleg",None,"validation"),("jxm/the_pile_subset",None,"train"),
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
