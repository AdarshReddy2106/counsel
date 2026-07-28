#!/bin/bash
#SBATCH --job-name=loadserver	   # Job name
#SBATCH --output=counsel.txt
#SBATCH --error=counsel_error.txt
#SBATCH -N 1
#SBATCH -p small
#SBATCH --cpus-per-task=48

# Load necessary modules
module load DL/conda-python/3.7
module load openmpi/4.1.4
module load gcc/10.2
module load gnu8/8.3.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate venv

# Change directory to where the server code is located
cd /home/aman.gupta/Thesis/Counsel/load-gen/

# Start fastapi servers in the background and capture their PIDs
uvicorn loadserver:app --host 0.0.0.0 --port 8000 --log-level warning &
PID1=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8001 --log-level warning &
PID2=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8002 --log-level warning &
PID3=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8003 --log-level warning &
PID4=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8004 --log-level warning &
PID5=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8005 --log-level warning &
PID6=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8006 --log-level warning &
PID7=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8007 --log-level warning &
PID8=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8008 --log-level warning &
PID9=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8009 --log-level warning &
PID10=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8010 --log-level warning &
PID11=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8011 --log-level warning &
PID12=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8012 --log-level warning &
PID13=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8013 --log-level warning &
PID14=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8014 --log-level warning &
PID15=$!
uvicorn loadserver:app --host 0.0.0.0 --port 8015 --log-level warning &
PID16=$!
# Change directory to where your Python scripts are located
cd /home/aman.gupta/Thesis/Counsel

# Run your Python scripts
python3 traina3c.py -n up16 -k 0.08 -nw 16 -e 500 &&
python3 traina3c.py -n op16 -k 0.02 -nw 16 -e 500 &&
python3 traina3c.py -n std16 -k 0.05 -nw 16 -e 500 &&

# Kill the uvicorn processes
kill $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12 $PID13 $PID14 $PID15 $PID16

# Optionally, you can uncomment the following lines to run additional scripts
# python3 traina3c.py -n std-f5-c3 -ncf 5 -ncp 3 -nw 5 -e 1000 &&
# python3 traina3c.py -n std-f10-c3 -ncf 10 -ncp 3 -e 1000 &&
# python3 traina3c.py -n std-f25-c3 -ncf 25 -ncp 3 -e 1000 &&
# python3 traina3c.py -n std-f50-c3 -ncf 50 -ncp 3 -e 1000 && 
# python3 traina3c.py -n std-f100-c3 -ncf 100 -ncp 3 -e 1000 
# python3 traina3c.py -n std-f5-c5 -ncf 5 -ncp 5 -e 1000
# python3 traina3c.py -n std-f5-c10 -ncf 5 -ncp 10 -e 1000
# python3 traina3c.py -n std-f5-c20 -ncf 5 -ncp 20 -e 1000

# Hyperparameter evaluation scripts
# python3 traina3c.py -n std01 -c 0.1 -k 0.05 -e 500
# python3 traina3c.py -n std02 -c 0.2 -k 0.05 -e 500
# python3 traina3c.py -n std03 -c 0.3 -k 0.05 -e 500
# python3 traina3c.py -n op01 -c 0.1 -k 0.02 -e 500
# python3 traina3c.py -n op02 -c 0.2 -k 0.02 -e 500
# python3 traina3c.py -n op03 -c 0.3 -k 0.02 -e 500
# python3 traina3c.py -n up01 -c 0.1 -k 0.07 -e 500
# python3 traina3c.py -n up02 -c 0.2 -k 0.07 -e 500
# python3 traina3c.py -n up03 -c 0.3 -k 0.07 -e 500
