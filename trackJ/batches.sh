# reproducible experiment launcher (each line = one background process)
cd "$(dirname "$0")"; export NT=2
B='"bits":3,"g":64,"clip":1'
A='"recipe":"A","bits":3,"g":128,"clip":1'
python tj.py "[{\"name\":\"w3 gptq\",$B},{\"name\":\"w3 gptq+bias\",$B,\"bias\":1},{\"name\":\"w3 gptq+bias+scale\",$B,\"bias\":1,\"scale\":1},{\"name\":\"w3 gptq+bias+scale+ln\",$B,\"bias\":1,\"scale\":1,\"ln\":1}]" > w3a.log 2>&1 &
python tj.py "[{\"name\":\"w3 gptq+rot256\",$B,\"rot\":256},{\"name\":\"w3 gptq+awq\",$B,\"awq\":1},{\"name\":\"w3 gptq+cle\",$B,\"cle\":1},{\"name\":\"w3 gptq+rot+bias+scale+ln\",$B,\"rot\":256,\"bias\":1,\"scale\":1,\"ln\":1}]" > w3b.log 2>&1 &
python tj.py "[{\"name\":\"A gptq\",$A},{\"name\":\"A +bias\",$A,\"bias\":1},{\"name\":\"A +bias+scale\",$A,\"bias\":1,\"scale\":1},{\"name\":\"A +bias+scale+ln\",$A,\"bias\":1,\"scale\":1,\"ln\":1}]" > Aa.log 2>&1 &
python tj.py "[{\"name\":\"A +rot256\",$A,\"rot\":256},{\"name\":\"A +awq\",$A,\"awq\":1},{\"name\":\"A +cle\",$A,\"cle\":1},{\"name\":\"A +rot+bias+scale+ln\",$A,\"rot\":256,\"bias\":1,\"scale\":1,\"ln\":1}]" > Ab.log 2>&1 &
python tj.py "[{\"name\":\"w2 g64 gptq\",\"bits\":2,\"g\":64,\"clip\":1},{\"name\":\"w2 +bias+scale+ln\",\"bits\":2,\"g\":64,\"clip\":1,\"bias\":1,\"scale\":1,\"ln\":1},{\"name\":\"w2 +rot+bias+scale+ln\",\"bits\":2,\"g\":64,\"clip\":1,\"rot\":256,\"bias\":1,\"scale\":1,\"ln\":1}]" > w2.log 2>&1 &
wait
