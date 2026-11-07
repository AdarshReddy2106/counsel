##README
## Steps I took to reproduce COUNSEL (Complete Debugged Workflow) Ubuntu Machine:

**1. Clone repository from Github**
* https://github.com/TheMonocledHamster/councel-a3c

**2. Ensure Docker Engine and Docker Compose are installed.**
* https://docs.docker.com/engine/install/
* https://docs.docker.com/compose/install/

**3. Pull ready-made images **
docker compose up ran into issues, so I used the following commands to pull ready images. This was suggested in the README of the above repository.
* 'docker pull themonocledhamster/counsel'
* 'docker pull themonocledhamster/load-gen'

**4. Configure `docker-compose.yaml`.**
* To ensure your services use host networking and the correct images(The ones pulled above), use the following as your updated final 'docker-compose.yaml':

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
 
  
  It was changed from:
  
  services:
  loadserver:
    image: load-gen
    build: ./load-gen
    container_name: loadserver
    expose:
      - "8000"
    ports:
      - "8000:8000"
    environment:
      - "LOADSERVER_PORT=8000"
      - "LOADSERVER_HOST=0.0.0.0"
  model:
    image: counsel
    build: .
    container_name: counsel
    network_mode: "host"
    

**5. Update the Dockerfile and train.py for Training and Inference (and Fix Command Chaining).**
* To set the default container behavior to run evaluation/inference instead of training, you must update the bottom of the 'Dockerfile'.
* Modify the 'Dockerfile' to comment out the training command and use the shell form (no brackets) for inference:

  # Comment out the default training command
  #CMD ["python3", "train.py", "-n", "std", "-k", "0.05", "-e", "1000"]

* *Directory Note:Ensure volume mounts and directory target variables (e.g., 'dir = 'data/'') are correctly updated from legacy 'a3c/' paths to unify where the trained models are saved and read from.(Line 12 in inference.py)

**6. Apply Pre-Training Code Patch.**
Issue encountered: The Python 3 container crashes with 'ModuleNotFoundError' during training due to legacy Python 2 relative imports.
* Open 'model/service_chain/chain.py', 'model/service_chain/component.py'.
* Add explicit dot notation to the imports (around line 11):
  ```python
  # Change from:
    from component import Component
    from state import State
  # To:
    from .component import Component
    from .state import State
  ```

**7. Initialize the Simulated Cloud Environment.**
The background traffic simulator must be running before any agents can train or evaluate. From the root directory, run:
* 'docker compose up -d loadserver'

**8. Execute the Training Process.**
* Run 'docker compose run model python3 train.py'.
* Note:For me, a run of the standard ('std') topologies on a CPU takes approximately 29 hours and successfully generates checkpoints in the 'data/' folder.

**9. Prepare Host Directory Permissions.**
Issue encountered: Docker volume mounting fails with '[Errno 13] Permission denied' when attempting to write inference logs back to the host machine.
* Manually create the log folder and grant permissions:
  ```bash
  mkdir infer_logs
  chmod 777 infer_logs
  ```
  This can be automated but this is what I did for now.

**10. Apply Inference Code Patches (Resolving Silent Failures).**
  ```
* **Patch B (Directory Hardcodes):** Open 'model/utils/test_policy.py'. The authors hardcoded the script to look for a 'model/' directory, but the PyTorch saves are in 'pyt_save/'. Fix this in both the directory checker and the loader:
  ```python
  # Line 19: pytsave_path = os.path.join(fpath, 'pyt_save')
  # Line 35: fname = osp.join(fpath,'pyt_save', 'model'+itr+'.pt')
  ```
* **Patch C (OpenAI Gym Mismatch):** In 'model/utils/test_policy.py' (Line 86), modern Gym environments return excess variables. Add a star operator ('*') to prevent a 'ValueError'.
  ```python
  # Change from: o, m, r, d, _ = env.step(a)
  # To: o, m, r, d, *_ = env.step(a)
  ```

**11.After Training, Run Inference & Fetch Data.**
All necessary training and inference will be run within the docker containers. You must use a volume mount to ensure the container utilizes your patched scripts instead of the broken native image scripts.
* 'docker compose run -v .:/counsel model python3 inference.py'
* The final metrics will be output to './infer_logs/inference.csv' and './infer_logs/time_tracker_std.csv'.(with the infer_logs folder that you have manually created.)
* At any time, 'bash copy_data.sh' can be run in to fetch output directly from a running container.
