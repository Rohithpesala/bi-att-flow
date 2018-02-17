# Getting BiDAF up for experimentation on Gypsum

0. Clone the repo.
```
git clone https://github.com/Rohithpesala/bi-att-flow.git
```
1. Cut a branch by your name
```
git checkout -b <name>
```
2. Push it to master.
```
git push origin <name>
```
3. Now, you are on your own branch. And any work you do will be your own.


## 0. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; .
```
```
/download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
> Note: If you're on gypsum, run `module load python/3.5.2` before you run the next command.
```
python -m squad.prepro
```

## 1. Submit training/testing/experiment job to Gypsum cluster

### Case 1: Single GPU - Not optimized.
This will run the experiment on a single gpu, with no optimization. Warning: Takes approximately 30 hours.

##### Step 1. Use the file `single-gpu.sh` in the folder slurm/single-gpu/.
```
cd slurm/single-gpu
```
##### Step 2: In the file `single-gpu.sh`, change line 30 to point to your `bi-att-flow` directory.

##### Step 3:
 - Uncomment line 35 to debug/run a small training experiment. Use this to see if your experiment is set up correctly.
 - Uncomment line 38 to start training.
 - Uncomment line 41 to start testing.

##### Step 4: Submit the job by running
 ```
 sbatch single-gpu.sh
 ```

 ### Case 2: Single GPU - Optimized.
 This will run the experiment on a single gpu, with optimizations enabled. Warning: May run out of working memory on the CPU after a couple of iterations and stop unpredictably mid-way.

 ##### Step 1. Use the file `optimized-single-gpu.sh` in the folder slurm/single-gpu/.
```
cd slurm/single-gpu
```
##### Step 2
Look at Case 1, step 2

##### Step 3
Look at Case 1, step 3

##### Step 4: Submit the job by running
 ```
sbatch optimized-single-gpu.sh
```

### Case 3: Multi-GPU - Optimized.
This will run the experiment on 2 gpus, with optimizations enabled. Note that here we are running the model only from

##### Step 1. Use the file `2-gpu.sh` in the folder slurm/multi-gpu/.
```
cd slurm/multi-gpu
  ```
  ##### Step 2
  Look at Case 1, step 2

  ##### Step 3
  Look at Case 1, step 3

  ##### Step 4: Submit the job by running
   ```
   sbatch 2-gpu.sh
   ```

### Submitting to and working with Gypsum
Once you have run your `sbatch` command, your job has been submitted to the gypsum job scheduler.
Take note of the job id.

You can also get the id of the job you just submitted if you run the command
```
squeue | grep <your user id>
```
where `<user id>` is your login id to slurm (everything before the @). Examples - usaxena, iankit, ruppaal ...
You'll see an output like:
```
JOBID	PARTITION	NAME	USER 		ST	  TIME  NODES	NODELIST(REASON)
3421260 	titanx-sh	bidaf	usaxena	R		0:00	1		 023
```
 > Note: The job ID's are strictly increasing. A job started now would have an ID number higher than a job which was started a second ago.

 When you submit your job to Gypsum, you're submitting to a job queue. And your job getting scheduled depends on the load of the cluster at the current time. So it might get queued and you'll see a `PD` written when you do an `squeue` under the column `status`.

```
JOBID	PARTITION	NAME	USER 		ST	  TIME  NODES	NODELIST(REASON)
3421160 	titanx-sh	nohup	rishikes	PD		0:00	1		 (PartitionTimeLimit)
```

To cancel the job, you can just use
```
scancel <SLURM_JOB_ID>
```

You can look at the different partitions by using the `sinfo` command.

```
PARTITION     AVAIL  TIMELIMIT   NODES     STATE 	NODELIST
matlab          	up      infinite      		1  			down* 	node100
m40-short        up    	 4:00:00      	2 		  drain* 	  node[001,014]
m40-short        up    	 4:00:00      	1  		 down* 	  node008
m40-short        up      4:00:00     	 15    			mix 	    node[002-007,009-010,012,015-019,025]
m40-short        up      4:00:00      	  7   			idle 		node[011,013,020-024]
m40-long         up     7-00:00:00        2 			drain* 		node[001,014]
```

Different partitions here mean different GPU and job length limit types.
A `short` name as `m40-short` or `titanx-short` signals that all jobs on this partition have a 4 hour limit, which can also be seen from the `sinfo` output.
A `long` name as `m40-long` or `titanx-long` signals that all jobs on this partition have a 7 day limit.

`titanx` and `m40` are the type of GPU's, with `m40`'s being better (faster, more memory), but are harder to get (#thatslife)

#### Setting the gpu type
At peak times it may be possible that some partitions are overloaded and all jobs are getting queued. Here you can try different partitions. At the top of your slurm file you'll see the line

```
#SBATCH --partition=m40-short
```
Different options are
- `m40-short`
- `m40-long`
- `titanx-short`
- `titanx-long`


#### Output Files
If you look at any of the slurm files, there is a line near the top of the file :
```
#SBATCH --output=bidaf-m40s-%A.out
#SBATCH --error=bidaf-m40s-%A.err
```
The `%A` is the job id. So if the job id is 3421260 the files produced by the job will be of the form `bidaf-m40s-3421260.out` and `bidaf-m40s-3421260.err`.

The `*.out` files usually have the result of the output stream, and the `*.err` files usually have the error stream.

### TODO:
- [ ] Writing your own slurm file
- [ ] Working on the GPU directly
- [ ] Working with saved models - uploading them to a common place
