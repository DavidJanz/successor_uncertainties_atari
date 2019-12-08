# Successor Uncertainties, Atari Experiments

This code allows for reproduction of the Atari experiments in https://arxiv.org/abs/1810.06530. Click [here](https://djanz.org/successor_uncertainties/tabular_code) for code to reproduce the tabular experiments.

To reproduce results, clone && pip install the requirements, then run
```
python3 run_atari.py --game Enduro
```
to train a Successor Uncertainties model with parameters as per the paper. This will output training information in tensorboard format to a subdirectory called logs. To obtain test scores, run
```
python3 /path/to/log_folder output_file.txt
```
The final score will be output to output_file.txt and progress of testing will be reported to stdout.
