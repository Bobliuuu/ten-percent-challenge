# DRAFT comment for the rules thread (issue #1) - NOT POSTED

Hi, and thank you for running this challenge. We want to follow the rules exactly as written, so we have a few yes/no questions about edge cases. Short answers are plenty.

1. Is it allowed to start from an already-compressed released model (for example a 1-bit or ternary 8B/27B release), with size measured against the fp16 size of the model we start from?
   Context: the rules say "10% of fp16", and we want to be sure which fp16 baseline applies.

2. Is a closed-form edit of existing weight tensors acceptable as "feedback without training"?
   Context: no gradient training, no new parameters, no side memory; for example a protected rank-limited update of attention `o_proj` computed once from 25 or fewer annotated items.

3. If the starting model is multimodal or larger than 0.5B, is its size counted by the whole-model fp16 bytes?
   Context: this includes every tensor in the checkpoint, not only the text decoder.

4. Is vocabulary trimming (removing token rows never used by the evaluation) and dropping an unused vision tower acceptable?
   Context: both only delete existing parameters; nothing is added or retrained.

5. Among entries that pass all rules, is "best" ranked by smallest size, by highest retention, or by something else?
   Context: knowing the tie-break lets us report the right number.

Thanks again for your time. Happy to adjust our approach to whatever you decide.
