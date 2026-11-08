import signal
from datasets import load_dataset
C=[("CFPB/consumer-complaints",None,"train"),("CFPB/consumer-finance-complaints",None,"train"),("NLBSE/nlbse23-issue-classification",None,"train"),("NLBSE/nlbse24-issue-report-classification",None,"train"),("lewtun/github-issues",None,"train"),
("mteb/stackexchange-clustering",None,"test"),("SetFit/amazon_counterfactual_en",None,"train"),("mteb/amazon_reviews_multi","en","train"),("tasksource/crowdflower","airline-sentiment",None),
("rungalileo/20_Newsgroups_Fixed",None,"train"),("SetFit/20_newsgroups",None,"train"),("fancyzhx/yelp_review_full",None,"train"),("McAuley-Lab/Amazon-Reviews-2023","raw_meta_Appliances","full"),
("vincentmin/eli5_rlhf",None,"train"),("ag_news_subset",None,"train"),("SetFit/bbc-news",None,"train"),("SetFit/sst2",None,"train"),("SetFit/enron_spam",None,"train"),
("Vezora/Mega-Phishing",None,"train"),("ealvaradob/phishing-dataset","combined_reduced","train"),("Anthropic/discrim-eval",None,"train"),("textdetox/multilingual_toxicity_dataset","en","en"),
("Paul/hatecheck",None,"test"),("ToxicityPrompts/RealToxicityPrompts",None,"train"),("jigsaw_unintended_bias",None,"train"),("thu-coai/diasafety",None,"train"),("allenai/real-toxicity-prompts",None,"train"),
("nlpaueb/legal_bert",None,"train"),("lex_glue",None,"train"),("coastalcph/lex_glue","eurlex","train"),("pile-of-law/pile-of-law","atticus_contracts","train"),
("jacobvs/PoliticalTweets",None,"train"),("launch/open_question_type",None,"train"),("Ramos-Ramos/emotion",None,"train"),("mteb/emotion",None,"train"),
("microsoft/ms_marco","v1.1","train"),("sentence-transformers/natural-questions",None,"train"),("BeIR/nfcorpus-qrels",None,"train"),
]
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM,h)
for n,cfg,sp in C:
    signal.alarm(30)
    try:
        d=load_dataset(n,cfg,split=sp,streaming=True) if sp else None
        r=next(iter(d)); print("OK ",n,cfg,{k:str(v)[:40] for k,v in list(r.items())[:6]},flush=True)
    except TO: print("TIMEOUT",n,flush=True)
    except Exception as e: print("FAIL",n,cfg,type(e).__name__,str(e)[:70].replace("\n"," "),flush=True)
    signal.alarm(0)
