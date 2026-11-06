exec(open('/Users/jhome/ten-percent/trackA/run7.py').read().split("for bits,g in")[0])
w0=base_model().bert.embeddings.word_embeddings.weight.data
U,S,Vh=torch.linalg.svd(w0,full_matrices=False)
for r in [128,192,256,384]:
    A=U[:,:r]*S[:r]; Bm=Vh[:r]
    for bA in [16,4]:
        m=base_model()
        if bA<16:
            qa,sa,za=quant_rtn(A,bA,64); A2=dequant(qa,sa,za,64) if False else dequant(qa,sa,za,64)
        else: A2=A
        m.bert.embeddings.word_embeddings.weight.data=A2@Bm
        by=(A.numel()*bA/8+(A.numel()/64*4 if bA<16 else 0)+Bm.numel()*2)/1e6
        r3f(m,f"FULL vocab SVD r{r} A@{bA}bit bytes={by:.1f}MB")
