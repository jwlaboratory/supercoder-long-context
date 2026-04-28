1. run experiment to see how supercoder does with long context from extended dataset
2. if it performs poorly, we know logn context is cooked


3. we can try to solve it using chunking:
-> shawn idea we basically decompose it somehow


4. we can try to solve it using 
-> we can try to solve it using lazy edits
-> morph LLM
-> we train RL to do lazy edits, then another model edits the main generated code

experiments:
 -> RL on base model with points for being 
 -> SFT first? so it knows the way it should do stuff?
 -> RL ontop of the supercoder model?# supercoder-long-context





files:

experiments/
1-experiment-supercoder-long/: this is the experiment for the supercoder model on 4 chunked lenghts of datasets. it proves that as C code increases, the model performes worse.
2-experiment-fast-apply/: this is the experiment to see if fast apply is viable by running with the lazy edit prompt. it show sthat supercoder cannot follow instructions prompts, but base qwen can, so this project is viable

training/
verl/: this has the copy pasted verl code from sueprcoder, its used as a framework for RL
train1-lazy-supercoder/: this is the training script for the lazy supercoder model: 
- we have a editprompt that takes the dataset and modifes to the prompt to include lazy edits prompt
-the model_train is the same as the original mode_train
-reward is the same, but it includes part that calls morph to fast apply first

benchmark/
merge_checkpoint
infer/plot : for testing the lazyedit model on the SuperCoder validation set
infer_long/plot_long : for testing the lazyedit model on different bucketed lengths of datasets

