experiment 1:
we basically compare supercoder and qwen-2.5-coder-7b-instruct on the long context dataset
we bucket the dataset into small med medlarge and large
we notice performance degrades as the context length increases


expiemernt 2:
we try to see if lazy edit is viable
we see that the lazy edits are "writing" into the base qwen, but not supercoder
good sign, we can trian on this

plan is to mix in dataset of long codenet stuff and also the original supercoder dataset
prompt includes lazy edit prompt
we eval by doing fast apply + testing it