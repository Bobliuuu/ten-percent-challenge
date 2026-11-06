# The 10% challenge

![The 10% challenge](challenge.png)

Small, just as good, and still learning.

**Prize: 1,000 USD** for the first complete, working proof.

## The challenge

1. **Start from one trained model.** Any model is fine, as long as it is above 0.5B parameters, either before or after you cut it. Pick a commercial-class task: anything commercial or of decent utility, nothing toy-shaped and nothing with a really narrow distribution. The model already does the job, and its score is the bar to meet.
2. **Keep about 10% of its fp16 size.** You can't train the weights. Every other technique is fair game.
3. **Match the base.** Your model gets the same score as the one you started from.
4. **Pass an exam it never saw.** Set aside a held-out, unseen subset before you begin. Keep it locked away until the very end, and never use it to build or to choose.
5. **Let it grow with user feedback.** Feedback means annotation or labelling, where someone marks the model's work, with very few samples, the way a real user would give it. It learns the new and keeps the old, still with no weight training, and it still passes the locked exam.

Submit your proof by [opening a GitHub issue](https://github.com/DeusExperiments/ten-percent-challenge/issues/new) on this repo by **12 November 2026, 17:05 UTC** (7 days from [the announcement](https://x.com/tokenbender/status/2118389310982643779)). Only the first person to submit a complete, working proof of beating the challenge qualifies for the prize.

How would you do it?
