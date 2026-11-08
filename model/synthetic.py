import importlib.util
import os
import sys
from typing import Any, List, Tuple

import numpy as np
import requests

from .service_chain.component import Component

_INPROC_SERVER: Any = None
_LoadSetter: Any = None
_BaseLoad: Any = None
_USE_INPROC: bool = False


def _get_inproc_server() -> Any:
    """Lazily initialize in-process LoadServer fallback when Docker load-gen is not running."""
    global _INPROC_SERVER, _LoadSetter, _BaseLoad
    if _INPROC_SERVER is None:
        load_gen_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "load-gen")
        )
        if load_gen_dir not in sys.path:
            sys.path.insert(0, load_gen_dir)
        loadserver_path = os.path.join(load_gen_dir, "loadserver.py")
        spec = importlib.util.spec_from_file_location("loadserver", loadserver_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load loadserver module from {loadserver_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _INPROC_SERVER = mod.LoadServer()
        _LoadSetter = mod.LoadSetter
        _BaseLoad = mod.BaseLoad
    return _INPROC_SERVER


def set_slo(slo: int, freq: int, knob: float, port: int = 8000) -> None:
    """
    Set the SLO for the load (uses HTTP load-gen if running, else in-process LoadServer).
    """
    global _USE_INPROC
    url = f"http://localhost:{port}/slo"
    query = {"slo": slo, "freq": freq, "knob": knob}
    try:
        resp = requests.put(url, json=query, timeout=0.5)
        resp.raise_for_status()
        _USE_INPROC = False
    except requests.RequestException:
        _USE_INPROC = True
        srv = _get_inproc_server()
        srv.set_load(_LoadSetter(slo=slo, freq=freq, knob=knob))


def set_base(comps: List[Component], port: int = 8000) -> None:
    """
    Set the base resources for the load.
    """
    cpu = [comp.cpu for comp in comps]
    mem = [comp.mem for comp in comps]
    if not _USE_INPROC:
        url = f"http://localhost:{port}/base"
        query = {"cpu": cpu, "mem": mem}
        try:
            resp = requests.put(url, json=query, timeout=0.5)
            resp.raise_for_status()
            return
        except requests.RequestException:
            pass
    srv = _get_inproc_server()
    srv.set_base(_BaseLoad(cpu=cpu, mem=mem))


def call_load_server(cpu: List[int], mem: List[int], port: int = 8000) -> Tuple:
    """
    Call the server with the action and get the next metrics.
    """
    if _USE_INPROC:
        metrics = _get_inproc_server().get_load()
    else:
        url = f"http://localhost:{port}/load"
        try:
            metrics = requests.get(url, timeout=0.5).json()
        except requests.RequestException:
            metrics = _get_inproc_server().get_load()
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
