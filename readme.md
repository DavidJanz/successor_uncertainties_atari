Code for paper "Successor Uncertainties: Exploration and Uncertainty in Temporal Difference Learning" by David Janz<sup>\*</sup>, Jiri Hron<sup>\*</sup>, Przemysław Mazur, Katja Hofmann, José Miguel Hernández-Lobato, Sebastian Tschiatschek. NeurIPS 2019.
<sup>\*</sup> Equal contribution

Paper is available at https://arxiv.org/abs/1810.06530.

This code allows for reproduction of the Atari experiments. Click [here](https://djanz.org/successor_uncertainties/tabular_code) for code to reproduce the tabular experiments.

To reproduce results, clone && pip install the requirements, then run
```
python3 run_atari.py --game Enduro
```
to train a Successor Uncertainties model with parameters as per the paper. This will output training information in tensorboard format to a subdirectory called logs. To obtain test scores, run
```
python3 /path/to/log_folder output_file.txt
```
The final score will be output to output_file.txt and progress of testing will be reported to stdout.
