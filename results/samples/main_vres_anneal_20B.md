# Samples: main_vres_anneal, 20B tokens (submitted model)

These come from the final weights of `main_vres_anneal`, the submitted model (49,822,228 parameters, 38,146 steps, 20B tokens). The prompts and settings are the same as in [`sweep_A_vres_1B.md`](sweep_A_vres_1B.md), so the two files can be read side by side.

Quality in one line: sentences are fluent and paragraphs hold their topic and register longer than at 1B, but facts are still invented or self-contradictory, it cannot do the arithmetic, and greedy decoding still loops.

Generated on an Apple M4 laptop (MPS) with `scripts/demo.py`, weights `main_vres_anneal/final.pt`, tokenizer `tok/tok32k/tokenizer.json`:

```bash
uv run python scripts/demo.py --ckpt final.pt --tok tokenizer.json \
    --max-new 80 --temperature 0.8 --top-k 50 --seed 0 --prompt "..." --prompt "..."
```

The six sampled prompts ran in one process in the order below, so each sample depends on the seed and on the prompts before it. Each was cut at 80 new tokens. Prompts are in bold. Output is copied unedited.

## Sampled (temperature 0.8, top-k 50)

### Factual sentence start

**The first person to walk on the Moon was** a 4th century BC man named Cressinus, according to a report from a Chinese expeditionary expedition to the Moon. During the 18th century, a number of Chinese explorers began to explore the Moon, and by the 19th century, China had become an important partner of the European exploration of the Moon.
The first man to walk on the Moon was a British astronomer, Henry

### Story opener

**Once upon a time, in a small village by the sea,** there was a man. But he was a man of a low stature. He was not one of the many things that made him unique. After years of living in small villages, the man’s stature and popularity have gradually declined. Now, despite his very recent return, he remains the most popular man in the village.
If you ask your friends who they are, these man are always asking: “

### Maths word problem

**Sarah has 12 apples. She gives 5 to her brother and buys 3 more. How many apples does she have now?** How many apples does she have left?
• ### How many apples can she buy at a loss?

How many apples can she buy at a loss? The price increases by 5% on this day. If the remaining apples were sold at 4% off of the original price, they would have cost 8 dollars today, and 3 dollars later, which is 3 times as much.

### Wikipedia-style lead

**The Amazon River is** one of the world’s largest rivers. It is also one of the longest rivers in the world. From its source in the Amazon region to reach its destination in the middle of the world, this river is home to many different creatures. The Amazon basin is where the Amazon River and its tributaries merge. The Amazon basin is the world’s largest river with its source in the Amazon region. As such

### How-to

**How to make a cup of tea: First,** it needs to be placed in the right place. Second, it needs to be placed in a container that is approximately the size of a pinhead. And finally, it needs to be completely wrapped in a cup that is approximately the size of a pinhead. When placed in a cup, the food will not be able to reach the plate. So, it is best to wrap it in a cup

### Dialogue

**"Where are you going?" asked Tom.** He was in the bathroom while the others were sitting, with the same answer for one of them. Tom didn’t even try to speak, and the person in the dark said something not very obvious. "I’m going to have a small group on the floor in front of you. I’m going to tell you about one of your friends. They are working on a math project for their class.

## Greedy (temperature 0)

**The Amazon River is** the largest river in the world, and the largest river in the world. It is the second largest river in the world after the Nile. The Amazon River is the second largest river in the world after the Nile. The Amazon River is the second largest river in the world after the Nile. The Amazon River is the second largest river in the world after the Nile. The Amazon River is the second largest

Greedy decoding contradicts itself in the second sentence and then loops, as the 1B checkpoint did.

## Speed

About 1.0-1.4 s per 80-token sample on MPS after the first (3.0 s including warm-up).
