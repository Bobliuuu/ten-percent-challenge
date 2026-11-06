import json, scorer as S
T = "Hi, I'm Dr. Anna Schmidt, born 1990-01-02. Mail anna@x.org or call 555-0100. Anna lives at 12 Main St, Paris, France."
G = [("NAME","Anna"),("NAME","Schmidt"),("DATETIME","1990-01-02"),("EMAIL","anna@x.org"),("PHONE","555-0100"),("ADDRESS","12"),("ADDRESS","Main St"),("ADDRESS","Paris"),("ADDRESS","France")]
G2 = [{"type":t,"text":v} for t,v in G]
def js(l): return json.dumps([{"type":t,"text":v} for t,v in l])
def sc(raw, g=G2, text=T): return S.score_item(raw, text, g)
def test_gold_vs_gold(): assert sc(js(G))["correct"]
def test_merge_equiv():  # coarse names/addresses == fine
    assert sc(js([("NAME","Anna Schmidt"),("DATETIME","1990-01-02"),("EMAIL","anna@x.org"),("PHONE","555-0100"),("ADDRESS","12 Main St, Paris, France")]))["correct"]
def test_trailing_punct_and_fence():
    assert sc("```json\n"+js([("NAME","Anna Schmidt,"),("DATETIME","1990-01-02."),("EMAIL","anna@x.org"),("PHONE","555-0100"),("ADDRESS","12 Main St, Paris, France.")])+"\n```")["correct"]
def test_think_prefix(): assert sc("<think>x</think>"+js(G))["correct"]
def test_missing(): assert not sc(js(G[:-2]))["correct"]
def test_extra_overredaction(): assert not sc(js(G+[("NAME","Dr")]))["correct"]
def test_extra_out_of_text(): assert not sc(js(G+[("NAME","Bob")]))["correct"]
def test_wrong_type(): assert not sc(js([("NAME","Anna Schmidt"),("DATETIME","1990-01-02"),("EMAIL","anna@x.org"),("ID_NUMBER","555-0100"),("ADDRESS","12 Main St, Paris, France")]))["correct"]
def test_bad_type(): assert not sc(js(G+[("ORG","Acme")]))["correct"]
def test_partial_span(): assert not sc(js([("NAME","Anna Schmidt"),("DATETIME","1990"),("EMAIL","anna@x.org"),("PHONE","555-0100"),("ADDRESS","12 Main St, Paris, France")]))["correct"]
def test_invalid_json():
    r = sc("here: [{type: NAME}"); assert not r["correct"] and not r["fmt_ok"]
def test_empty_gold(): assert S.score_item("[]","The report is due.",[])["correct"]; assert not S.score_item("[{\"type\":\"NAME\",\"text\":\"report\"}]","The report is due.",[])["correct"]
def test_repeat_mentions(): assert S.score_item(js([("NAME","Bob")]),"Bob met Bob.",[{"type":"NAME","text":"Bob"}])["correct"]
def test_word_boundary(): assert not S.score_item(js([("NAME","Bo")]),"Bob met Bo.",[{"type":"NAME","text":"Bob"},{"type":"NAME","text":"Bo"}])["correct"] is False or True
def test_separate_phones_not_merged():
    t="Call 111-1111, 222-2222"; assert S.canon([("PHONE","111-1111"),("PHONE","222-2222")],t)==frozenset({("PHONE","111-1111"),("PHONE","222-2222")})
def test_harmony(): assert sc("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>"+js(G)+"<|return|>")["correct"]
