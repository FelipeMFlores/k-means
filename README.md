# k-means
Project for Machine Learning class. Developing k-means.

# Installation instructions

## Requirements
- Python 3
- Linux / MacOS (the instructions may be adapted for Windows usage)

## Installing dependencies

To install all required dependencies from the `requirements.txt` file, run the following command:

```shell
python3 -m pip install -r requirements.txt
```

# Running the Random Forests algorithm

To run the k-means algorithm, you may simply run:

````shell
pip3 main.py
````

The value of k for each experiment may be changed on the variable `k`, which is found on line 53 of the `main.py` file. 

Similarly, you may uncomment/comment lines 65 to 69 to run each hypothesis.A description of the hypothesis in question can be found on the file `hypothesis.py`.

By uncommenting lines 59 through 63, you may run the simulation with values of k rangin from 1 to 12, to analyze the graph and apply the elbow method to find the best value of k.
