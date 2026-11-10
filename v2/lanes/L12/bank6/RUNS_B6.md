# L12 on BANK6 runs (Modal app tenpct-l12-b6, volume tenpct-l12-b6)
- 2026-11-09 launched (A100-80GB each): exam1 fc-01M71M19BTTBMZCZRJDFTZCFWR (locked exam, RTN4 mass k25, pre-declared), sel1 fc-01M71M1A0STT0G6ZKM3ASWDAQY (SEL config sweep: rtn4 k25/k35, gptq+had 4/3-bit), fbdev1 fc-01M71M1AN7NX5ZH7W3K0R3YGMM, fbdev2 fc-01M71M1BA7CP90089P4E8RRCWB (feedback dev grid, DEV data only, 698 cal rows)
- gdiag1 fc-01M71MYXWWEE60A45CJNET3K43 (engine GPTQ NaN/garbage diagnosis; SEL sweep sel1 done: rtn4 k25 .961, k35 .977 (9.98% over cap), all engine-gptq configs acc 0)
- sel2 (gptq+had 4-bit k25/k30), sel3 (gptq+had 3-bit k33/k40) with non-finite retry/RTN fallback patch (gdiag1 showed 8 of 64 experts/layer non-finite in engine gptq_b, 62/64 in layer 0 gate/up)
