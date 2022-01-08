In the attached code, we just run one replication to show results. Thus, the results are slightly different from those in the paper (100 replications).
You just need to
(a) Run "main.py" to get the estimates of the parameters.
(b) Run "analysis.py" to show the results.


Details about the attached files:
(1) "data.py" is the code for generating simulation data. You can set arg "Rep" to adjust the replication number.
Here we set "Rep =1" to save time. The generated data is stored in ".npy".
Make sure that your computer has enough space because SVD requires time and space.
In our experiment, we run the code in the cluster and we attached the generated data in the folder in case your computer fails.

(2) "main.py" is the main code to obtain the estimates in all iterations.
After running it, the results are stored in ".npy".

(3) "update.py" and "tool.py" include the functions to update the parameters in MCMC algorithm.

(4) "analysis.py" is to show the Bias and RMS of the parameters and show the other estimates, such as, the estimated eigenimage and trajectory function in trajectory model (4).
