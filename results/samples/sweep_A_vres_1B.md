# Samples: sweep_A_vres, 1B tokens

These come from the `sweep_A_vres` checkpoint of the 1B-token ablation sweep (49,822,228 parameters, 1,907 steps), **not** the final 20B-token model. They show what the demo does and roughly where the recipe stands after 1B tokens.

Quality in one line: the text is grammatical and keeps the register of the prompt (story, encyclopedia, worksheet, dialogue), but facts are wrong, it repeats itself, and it does not do arithmetic.

Generated on an Apple laptop (MPS) with `scripts/demo.py`, weights `sweep_A_vres/final.pt`, tokenizer `tok/tok32k/tokenizer.json`:

```bash
uv run python scripts/demo.py --ckpt final.pt --tok tokenizer.json \
    --max-new 80 --temperature 0.8 --top-k 50 --seed 0 --prompt "..." --prompt "..."
```

The six sampled prompts ran in one process in the order below, so each sample depends on the seed and on the prompts before it. Each was cut at 80 new tokens. Prompts are in bold.

## Sampled (temperature 0.8, top-k 50)

### Factual sentence start

**The first person to walk on the Moon was** a 4-year-old girl who was born on the moon. But her father died a few days later. Her father died on the moon. He could not walk on the Moon, because he couldn't walk on the earth.
The story of the Moon was written in 1858. At the time of the first walk, a boy was born on the moon. The boy is said to have

### Story opener

**Once upon a time, in a small village by the sea,** there was a high mountain on the river. There were a thousand people, many of them in the same family. The village of Kupka was one of the small villages in the valley, which was a place of pilgrimage and relaxation for the local people. The village was a village church where the people were invited to perform a special service by men, who were able to carry out their work as

### Maths word problem

**Sarah has 12 apples. She gives 5 to her brother and buys 3 more. How many apples does she have now?** How many apples do she have left in her store? How many apples in a year are the three most important apples?

```
### Year 1

##### Stage: 2 Challenge Level:

Can you work out how many different apples there are in a package and what happens to them together?

### Sorting

##### Stage: 2 and 3

An easy way to organize your collection of data is
```

### Wikipedia-style lead

**The Amazon River is** one of the world's most important rivers. It's one of the longest rivers on earth - it's 7.4 million miles (3.3 million km) long. The most famous rivers of its kind are the Amazon River (the largest river on earth), the Amazon River (the largest river in the world), the Peruvian Amazon (the biggest river in the world), the Congo River

### How-to

**How to make a cup of tea: First,** you need to make a cup of tea by carefully reading the label on the label. Then, you should get the first name of the drink as you are going to make the first name of the coffee. Then you should get the second name of the drink as you go. Next, make a cup of tea by carefully reading the label on the label.
Once the cup has been made, you can

### Dialogue

**"Where are you going?" asked Tom.**
**"**Where are you going?" asked the others.
"At the very beginning, there was a new thing going on inside the room, and there was a little, quite a bit of something, going on within the room," said Tom.
"At the end, there was something new. As it turns out, there's one new thing in the room, one new thing that's been there before."

## Greedy (temperature 0)

**The Amazon River is** a river that flows through the Amazon River. It is the largest river in the world, and the largest in the world. It is the largest river in the world, and the largest in the world. It is the largest river in the world, and the largest in the world. It is the largest river in the world, and the largest in the world. It is the largest river in the

Greedy decoding falls into a loop within two sentences, which is typical for a small model.

## Speed

About 1-2 s per 80-token sample on MPS after the first (4.6 s including warm-up). A CPU run (`--device cpu`, 30 tokens) also worked.
