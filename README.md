# ArithmeticLLM
Code for "Principled Understanding of the Generalization of Transformer Models to Arithmetic Reasoning", (2024)

Remark: The code is cloned from https://github.com/karpathy/nanoGPT. We have made modifications based on the code and our arithmetic tasks.


# Use Cases

## Generate data

Utilize the code located in the "data" folder to produce the task data.

## Prepare data

For Addition and Modular Addition:

```sh
python data/addition/prepare_linebyline.py
```

For Addition and Modular Addition:

```sh
python data/multiply/prepare_linebyline.py
```

This creates a `meta.pkl`, `train.pkl` and `val.pkl` in that data directory.

## Model Training

Use GPU for additon/modular addtion training:

```sh
python train.py config/train_addition_char.py \
--tensorboard_log=True \
--run_id=1 \
--device=cuda:0 \
--line_train=True \
--ckpt_name=ckpt_add.pt
```

Use GPU for multiplication/modular multiplication training:

```sh
python train.py config/train_multiply_char.py \
--tensorboard_log=True \
--run_id=1 \
--device=cuda:0 \
--line_train=True \
--ckpt_name=ckpt_multiply.pt
```
