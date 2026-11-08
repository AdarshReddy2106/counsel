# COUNSEL: Complete Setup & Madhava GPU Cluster Reproduction Guide

This guide documents all code fixes, offline dependency packaging, and SLURM cluster commands required to reproduce COUNSEL training (`train.py`) and evaluation (`inference.py`) locally with Docker and on the **IIT Palakkad Madhava HPC Cluster**.

---

## 1. Required Repository Code Fixes (One-Time Local Setup)

Before running locally or packaging for the cluster, ensure these 7 files in `D:\counsel` have the following fixes applied:

### 1.1 `docker-compose.yaml` (Pre-built Images & Host Networking)
```yaml
services:
  loadserver:
    image: themonocledhamster/load-gen
    container_name: loadserver
    network_mode: "host"
    environment:
      - "LOADSERVER_PORT=8000"
      - "LOADSERVER_HOST=0.0.0.0"
  model:
    image: themonocledhamster/counsel
    container_name: counsel
    network_mode: "host"
```

### 1.2 `Dockerfile` (Lines 28–31)
Comment out the default training `CMD` and use shell form for evaluation:
```dockerfile
#CMD ["python3", "train.py", "-n", "std", "-k", "0.05", "-e", "1000"]

CMD bash model_eval.sh && python3 inference.py
```

### 1.3 `inference.py` (Line 24)
Point inference to `data/` instead of legacy `a3c/`:
```python
    dir = 'data/'
```

### 1.4 `model/service_chain/chain.py` & `model/service_chain/component.py`
Fix Python 2 relative imports for Python 3:
- In `model/service_chain/chain.py` (Lines 11–12):
  ```python
  from .component import Component
  from .state import State
  ```
- In `model/service_chain/component.py` (Line 8):
  ```python
  from .state import State
  ```

### 1.5 `model/ppo.py` (Line 305)
Unpack variable-length return tuple from `CloudEnv.step(a)` (which returns 7 values):
```python
            next_o, m, r, d, *_ = env.step(a)
```

### 1.6 `model/utils/test_policy.py` (Lines 19, 52 & 86)
Fix the checkpoint subdirectory name (`pyt_save`) and `env.step(a)` unpacking:
```python
# Line 19:
    pytsave_path = osp.join(fpath, 'pyt_save')

# Line 52:
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')

# Line 86:
        o, m, r, d, *_ = env.step(a)
```

### 1.7 `train.py` (Lines 76–89)
Disable `torch.autograd.profiler.profile(record_shapes=True, profile_memory=True)` so it does not exhaust memory during `update()`:
```python
roboconf.train(algo)
# start_time = time.time()
# with profiler.profile(record_shapes=True, profile_memory=True) as prof:
#     roboconf.train(algo)
# training_time = time.time() - start_time
# table_output = prof.key_averages().table(sort_by="cuda_time_total")
# 
# # Define the file name
# file_name = "profiler_output.txt"
# 
# # Open the file in write mode and save the output
# with open(file_name, 'w') as file:
#     file.write(f"Tensor Profiler (Training Time = {training_time})\n")
#     file.write(table_output)
```

---

## 2. Optional: Local 2-Epoch Sanity Check with Docker (Windows PC)

Ensure **Docker Desktop** is running with **Enable host networking** turned on (*Settings → Resources → Network*).

```powershell
cd D:\counsel

# 1. Pull pre-built images
docker pull themonocledhamster/counsel
docker pull themonocledhamster/load-gen

# 2. Start background traffic simulator on port 8000
docker compose up -d loadserver

# 3. Run 2-epoch test (default in model/configs/hyperparams.json)
docker compose run --remove-orphans -v .:/counsel model python3 train.py

# 4. Run inference test
docker compose run --remove-orphans -v .:/counsel model python3 inference.py
```

---

## 3. Packaging Code & Offline Wheels on Your Windows PC

Because Madhava login and compute nodes do not have outbound internet access, package both the code (`counsel.tar.gz`) and the Linux Python 3.9 wheels (`wheels.tar.gz`) on your PC and upload them via `scp` on port `49122`.

Run in **Windows PowerShell**:

```powershell
cd D:\counsel

# 1. Create code archive
tar -czvf counsel.tar.gz Dockerfile docker-compose.yaml train.py traina3c.py inference.py model_eval.sh copy_data.sh requirements.txt model load-gen plotters

# 2. Download Linux Python 3.9 wheels (only needed if wheels.tar.gz does not already exist)
mkdir -Force D:\counsel\wheels
pip download --dest D:\counsel\wheels --platform manylinux2014_x86_64 --python-version 39 --only-binary=:all: scipy==1.10.1 fastapi==0.87.0 uvicorn==0.20.0 requests==2.25.1 psycopg2-binary==2.9.5 joblib==1.2.0 psutil==5.9.3 tqdm==4.64.1 cloudpickle gym_notices sniffio exceptiongroup importlib-metadata zipp
pip wheel --no-deps --wheel-dir D:\counsel\wheels gym==0.26.2
tar -czvf wheels.tar.gz wheels

# 3. Upload both archives to Madhava (Port 49122)
scp -P 49122 D:\counsel\counsel.tar.gz b102301018-abhinandansp@madhava.iitpkd.ac.in:~/
scp -P 49122 D:\counsel\wheels.tar.gz b102301018-abhinandansp@madhava.iitpkd.ac.in:~/
```

---

## 4. First-Time Setup on Madhava Cluster

### 4.1 SSH into Madhava Login Node
From **Windows PowerShell**:
```powershell
ssh -p 49122 b102301018-abhinandansp@madhava.iitpkd.ac.in
```

### 4.2 Extract Archives in `/scratch/$USER/counsel` & Install Offline Packages
Run in the **Madhava terminal**:

```bash
# 1. Create project directory in /scratch and extract code + wheels
mkdir -p /scratch/$USER/counsel
mv ~/counsel.tar.gz ~/wheels.tar.gz /scratch/$USER/counsel/
cd /scratch/$USER/counsel
tar -xzvf counsel.tar.gz
tar -xzvf wheels.tar.gz

# 2. Activate Madhava's pre-installed GPU PyTorch environment (Python 3.9.21)
module load anaconda3-2024
source /home/apps/compilers/anaconda3/2024/bin/activate
conda activate pytorch_envgpu

# 3. Install all offline wheels into ~/.local
pip install --user --no-index --find-links=wheels gym scipy fastapi uvicorn requests psycopg2-binary joblib psutil tqdm cloudpickle

# 4. Ensure no broken OpenMPI C-extension exists and create the pure-Python single-process mpi4py shim
#    (Bypasses missing libpmix.so.2 on gpu01 since hyperparams.json uses threads=1)
rm -rf ~/.local/lib/python3.9/site-packages/mpi4py*
cat << 'EOF' > /scratch/$USER/counsel/mpi4py.py
class _Comm:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def Bcast(self, x, root=0): pass
    def Allreduce(self, sendbuf, recvbuf, op=None): recvbuf[:] = sendbuf

class MPI:
    COMM_WORLD = _Comm()
    SUM = 1
    MIN = 2
    MAX = 3
EOF
```

---

## 5. Creating `submit.sh` & Running the 1,000-Epoch Job on Madhava

Run in the **Madhava terminal** (`/scratch/$USER/counsel`):

```bash
cd /scratch/$USER/counsel

cat << 'EOF' > /scratch/$USER/counsel/submit.sh
#!/bin/bash
#SBATCH --job-name=COUNSEL_Job
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /home/apps/compilers/anaconda3/2024/bin/activate
conda activate pytorch_envgpu
export PATH="$HOME/.local/bin:$PATH"

mkdir -p data logs infer_logs
chmod 777 infer_logs

# 1. Start the background loadserver on port 8000
cd $SLURM_SUBMIT_DIR/load-gen
python3 -m uvicorn loadserver:app --host 0.0.0.0 --port 8000 --log-level warning &
LOADSERVER_PID=$!
cd $SLURM_SUBMIT_DIR

trap "kill $LOADSERVER_PID 2>/dev/null" EXIT
sleep 5

# 2. Run 1000-epoch training (-u for real-time unbuffered logging)
python3 -u train.py -n std -k 0.05 -e 1000

# Copy saved checkpoint so inference.py finds it under data/std-f5-c3/pyt_save
mkdir -p data/std-f5-c3
cp -r data/std/std_s0/pyt_save data/std-f5-c3/

# 3. Run inference
python3 -u inference.py
EOF

chmod 755 submit.sh
sbatch submit.sh
```

> **Quick Re-Run Anytime Later:**
> Once Step 4 & Step 5 are already in `/scratch/$USER/counsel`, any time you log in later to re-run training from scratch, simply run:
> ```bash
> cd /scratch/$USER/counsel
> sbatch submit.sh
> ```
> To train the other provisioning configurations (`op` or `up`), edit `python3 -u train.py -n std -k 0.05 -e 1000` in `submit.sh`:
> - Over-provisioning (`op`): `python3 -u train.py -n op -k 0.03 -e 1000`
> - Standard (`std`): `python3 -u train.py -n std -k 0.05 -e 1000`
> - Under-provisioning (`up`): `python3 -u train.py -n up -k 0.07 -e 1000`

---

## 6. Monitoring Training Midway & Fetching Results

### 6.1 Check Job Status & Live Logs on Madhava
Replace `<JOBID>` with the job number printed by `sbatch submit.sh`:

```bash
# 1. Check whether job is Running (R) or Pending (PD)
squeue --me

# 2. Watch live epoch-by-epoch training output (press Ctrl+C to stop watching)
tail -f <JOBID>.out

# 3. Check standard error (Note: the Gym deprecation warning at the top is harmless)
cat <JOBID>.err

# 4. Inspect saved epoch metrics and latest model checkpoint midway
tail -n 20 /scratch/$USER/counsel/data/std/std_s0/progress.csv
ls -lh /scratch/$USER/counsel/data/std/std_s0/pyt_save/model.pt

# 5. Check elapsed job runtime
sacct -j <JOBID> --format=JobID,JobName,Partition,State,Elapsed

# 6. Cancel a running job (only if needed)
scancel <JOBID>
```

### 6.2 Copy Trained Models & Logs Back to Your Windows PC

**Option A: Bundle into `results.tar.gz` on Madhava & Download in One Step (Recommended)**
1. In your **Madhava terminal**:
   ```bash
   cd /scratch/$USER/counsel
   tar -czvf results.tar.gz data logs infer_logs <JOBID>.out <JOBID>.err
   ```
2. In **Windows PowerShell**:
   ```powershell
   cd D:\counsel
   scp -P 49122 b102301018-abhinandansp@madhava.iitpkd.ac.in:/scratch/b102301018-abhinandansp/counsel/results.tar.gz D:\counsel\
   tar -xzvf results.tar.gz
   ```

**Option B: Copy Individual Folders Directly from Windows PowerShell**
```powershell
scp -P 49122 -r b102301018-abhinandansp@madhava.iitpkd.ac.in:/scratch/b102301018-abhinandansp/counsel/data D:\counsel\
scp -P 49122 -r b102301018-abhinandansp@madhava.iitpkd.ac.in:/scratch/b102301018-abhinandansp/counsel/logs D:\counsel\
scp -P 49122 -r b102301018-abhinandansp@madhava.iitpkd.ac.in:/scratch/b102301018-abhinandansp/counsel/infer_logs D:\counsel\
```

