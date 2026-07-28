import requests
import numpy as np
from typing import List, Tuple
import time
from .service_chain.component import Component


def set_slo(slo:int, freq:int, knob:float, port=8000)->None:
    """
    Set the SLO for the load.
    """
    url = "http://localhost:"+str(port)+"/slo"
    query = {"slo": slo, "freq": freq, "knob": knob}
    requests.put(url, json=query)

def set_base(comps:List[Component], port = 8000)->None:
    """
    Set the base resources for the load.
    """
    cpu = [comp.cpu for comp in comps]
    mem = [comp.mem for comp in comps]
    url = "http://localhost:"+str(port)+"/base"
    query = {"cpu": cpu, "mem": mem}
    requests.put(url, json=query)


def call_load_server(cpu:List[int], mem:List[int], port=8000)->Tuple:
    """
    Call the server with the action and get the next metrics.
    """
    url = "http://localhost:"+str(port)+"/load"
    metrics = requests.get(url).json()
    arrival_rate = metrics["arrival_rate"]
    lcpu = np.array(metrics["load"][0])
    lmem = np.array(metrics["load"][1])
    slo = metrics["slo"]
    done = metrics["done"]

    act_type = 0
    act_comp = -1
    flag = False
    # -->
    for i in range(len(cpu)):
        if lcpu[i] * 0.85 > cpu[i] or lmem[i] * 0.85 > mem[i]:
            flag = True
            act_type = 1
            act_comp = i
        elif lcpu[i] < cpu[i]*0.4 and lmem[i] < mem[i]*0.4 and not flag:
            flag = True
            act_comp = i
    
    loadc = [lcpu[i]/(cpu[i]+1e-7) for i in range(len(cpu))]
    loadm = [lmem[i]/(mem[i]+1e-7) for i in range(len(mem))]

    latency_factor = max(max(loadc), max(loadm),0)
    
    loadc = [min(1,i) for i in loadc]
    loadm = [min(1,i) for i in loadm]

    latency = slo * latency_factor
    
    return arrival_rate, loadc, loadm, latency, act_type, act_comp, done
