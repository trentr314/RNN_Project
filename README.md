# Recurrent Neural Networks in Stock Price Prediction
For the semester project in a machine learning class, I chose the topic of time-series prediction with a recursive neural network, and decided to use IBM stock price history as the time series data.

# Introduction
A project report PDF is included here which contains the underlying calculus, among other things.

If you would like to run the program in your browser, visit [this Repl](https://replit.com/@trentr314/stocksRNN?v=1) (please be patient as it takes a minute to run).

# Getting started
If you would like to run it on your own machine:

- If you do not have Python you need to [install it](https://www.tutorialsteacher.com/python/install-python).

- If you are on Windows, I recommend [Git Bash](https://gitforwindows.org/), which will allow you to use the same commands as on Mac or Linux.

- If you are on Windows using Git Bash, there is a trick to [using the python command in Git Bash](https://itecnote.com/tecnote/python-not-working-in-the-command-line-of-git-bash/).

# Requirements
The [Alpha Vantage API](https://www.alphavantage.co/) was used to retrieve stock price data.  I have publicly hosted the data I used on GitHub, so the .py file will access it from any machine with internet access.  You do not need to request any data from the API yourself.  However, for reference, other/alpha_vantage.py does contain the code I used to retrieve the data.

Modules that require installation (as in requirements.txt):
- matplotlib
- numpy
- pandas

Use the package manager **pip** to install
```bash
$ pip install matplotlib
```
```bash
$ pip install numpy
```
```bash
$ pip install pandas
```

# Configuration
Hyperparameters are hard-coded in lines 42-59.  They include
- hidden layer size

  The number of nodes in the hidden layer.  Set to 6 by default.
- learning rate

  The learning rate is 3\*10^-11.  I found this to be a good balance.  If you make it too small, say x\*10^-15, backpropogation will have a tiny, insignificant effect on the weights, and the algorithm won't really learn anything.  If you make it too large, say x\*10^-5, the algorithm will fly all over the place, constantly overcorrecting and never converging.
- number of epochs

  The number of passes the algorithm makes over the entire dataset.  I have this set to 200 by default.  Stock prices are not easily predicted and this algorithm needs to go through the dataset many times to learn significantly.
- batch size

  How many data points (days) are used in forward predictions before the algorithm performs backpropogation to refine the weights in the net.  Note that since the first iteration of each batch has no previous prediction, the actual input is treated as a prediction as well as an input, as if the previous prediction was perfect.  Larger batch sizes make this imperfection less significant.
- epoch size

  How much of the dataset is used.  The dataset retrieved was cropped down a bit to make a neat number of data points that was easily divisible by my chosen batch size (10).  (Dataset size) - (epoch size) = the number of the earliest data points that are cropped off of the beginning
- number of batches

  Be careful modifying epoch size and batch size, because the number of batches is calculated as int(epoch_size/batch_size).  Epoch size must be a multiple of batch size or this will cause problems.

# Usage
Run as any Python program in bash with `python3 RNN_project.py`

# Other notes
Output will occasionally overflow and give results of NaN.  This happens somewhat randomly due to random initialization of weights.  This is most likely to happen if you choose an unnecessarily high hidden layer size or a learning rate that is too high.  If this happens, please just be patient and try it a couple more times.

One of this program's weaknesses is its inability to predict more than 1 day in the future.  More advanced techniques would have to be implemented to predict multiple timesteps of data without the actual value of the previous timestep.  This prediction of multiple timesteps is a potential subject of future work.

As mentioned in Requirements, for reference, other/alpha_vantage.py contains the code I used to retrieve the data from the Alpha Vantage API.

# Contact
Trent Rogers

Email: trentr314@gmail.com
