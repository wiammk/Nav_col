"""Conservative prioritized space-time planning under this simulator's rules.

Plans use weighted movement costs and a 0.1 wait cost. Priorities are ascending
initial weighted distance, then robot index. Unplanned starts are temporarily
treated as permanent obstacles during higher-priority planning. Plans reserve
completed targets indefinitely, disallow swaps, and obey the current-occupancy
mask and stationary-robot rule. At most nine consecutive waits are allowed.
Node reservations are exclusive, even when the simulator permits capacity >1.
Replan after closures or unexpected execution. This is not an optimal CBS solver.
"""
import argparse
import heapq
import itertools
import json
import pickle
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import networkx as nx
import numpy as np
from evaluation.common_protocol import load_protocol, summarize
from scripts.experiments.review_followup_worker import factory, protocol_paths, evaluate, write_csv


def at(path,t):
    return path[min(t,len(path)-1)]


def space_time_path(graph,start,goal,reservations,blocked,horizon,stuck,stuck_limit=10):
    available=nx.subgraph_view(graph,filter_node=lambda n:n not in blocked or n==start)
    if start not in available or goal not in available:
        return None
    distances=nx.single_source_dijkstra_path_length(available,goal,weight='weight')
    if start not in distances:
        return None
    final_nodes={p[-1] for p in reservations}
    if goal in final_nodes:
        return None
    steady_graph=nx.subgraph_view(available,filter_node=lambda n:n not in final_nodes)
    steady_reachable=nx.node_connected_component(steady_graph,goal)
    reservations_settle=max((len(p)-1 for p in reservations),default=0)
    serial=itertools.count()
    first=(start,0,stuck)
    queue=[(distances[start],0.,next(serial),first)]
    costs={first:0.};parents={}
    while queue:
        _,cost,_,state=heapq.heappop(queue)
        u,t,waits=state
        if cost != costs.get(state):
            continue
        if t>=reservations_settle and u not in steady_reachable:
            if u not in final_nodes or not any(v in steady_reachable for v in available.neighbors(u)):
                continue
        if u==goal and all(goal not in p[t+1:] and p[-1]!=goal for p in reservations):
            result=[u]
            while state in parents:
                state=parents[state];result.append(state[0])
            return list(reversed(result))
        if u==goal:
            # First arrival terminates the robot; it cannot leave its goal again.
            continue
        if t>=horizon:
            continue
        old=[at(p,t) for p in reservations]
        nxt=[at(p,t+1) for p in reservations]
        for v in [*sorted(available.neighbors(u)),u]:
            waiting=v==u
            if waiting and waits+1>=stuck_limit:
                continue
            if waiting and any(a==u and b==u for a,b in zip(old,nxt)):
                # The existing pairwise swap check also flags co-waiting active
                # robots at a shared node. Match the evaluated implementation.
                continue
            if not waiting:
                if v in nxt:
                    continue
                cap=max(1,int(graph.nodes[v].get('capacity',1)))
                if old.count(v)>=cap:
                    continue
                if any(b==v and a==b for a,b in zip(old,nxt)):
                    continue
                arrivals=sum(b==v and a!=b for a,b in zip(old,nxt))
                if arrivals+1>cap:
                    continue
                if any(a==v and b==u for a,b in zip(old,nxt)):
                    continue
            incoming=sum(b==u and a!=b for a,b in zip(old,nxt))
            if incoming:
                # A higher-priority arrival must also pass its occupancy mask.
                cap_u=max(1,int(graph.nodes[u].get('capacity',1)))
                if waiting or old.count(u)+1>=cap_u:
                    continue
            next_state=(v,t+1,waits+1 if waiting else 0)
            step_cost=.1 if waiting else float(graph[u][v].get('weight',1.))
            total=cost+step_cost
            if total < costs.get(next_state,float('inf')):
                costs[next_state]=total;parents[next_state]=state
                heapq.heappush(queue,(total+distances.get(v,float('inf')),total,next(serial),next_state))
    return None


class PrioritizedPlanner:
    def __init__(self,env):
        self.env=env
        self.envs=env.envs if hasattr(env,'envs') else [env]
        self.key=None;self.paths=None;self.origin_step=0
        self.replans=0;self.failed_plans=0;self.mask_violations=0

    def signature(self):
        e=self.envs[0]
        return (tuple(x.target_node for x in self.envs),frozenset(e.closed_edges),frozenset(e.blocked_nodes))

    def replan(self):
        envs=self.envs;graph=envs[0].graph.copy()
        graph.remove_nodes_from(envs[0].blocked_nodes)
        graph.remove_edges_from([tuple(e) for e in envs[0].closed_edges])
        graph.remove_edges_from([(u,v) for u,v,d in graph.edges(data=True)
            if not d.get('passable',True) or float(d.get('width',1.))<envs[0].rcfg['min_door_width']])
        active=[i for i,e in enumerate(envs) if not e._terminated and not e._truncated]
        distances={i:nx.shortest_path_length(graph,envs[i].current_node,envs[i].target_node,weight='weight')
            if nx.has_path(graph,envs[i].current_node,envs[i].target_node) else float('inf') for i in active}
        priority=sorted(active,key=lambda i:(distances[i],i))
        fixed={i for i in range(len(envs)) if i not in active}
        # If a robot cannot plan, higher-priority paths already avoid its start.
        paths={i:[envs[i].current_node] for i in fixed}
        for offset,i in enumerate(priority):
            e=envs[i]
            unplanned={envs[j].current_node for j in priority[offset+1:]}
            unplanned.update(envs[j].current_node for j in fixed)
            path=space_time_path(graph,e.current_node,e.target_node,list(paths.values()),
                unplanned,e.max_steps-e._step_count,e._stuck_count,int(e.rcfg['stuck_limit']))
            if path is None:
                path=[e.current_node];fixed.add(i);self.failed_plans+=1
            paths[i]=path
        self.paths=paths;self.origin_step=max(e._step_count for e in envs)
        self.key=self.signature();self.replans+=1

    def action(self,index,obs):
        step=max(e._step_count for e in self.envs)
        t=step-self.origin_step
        expected=self.paths is not None and all(at(self.paths[i],t)==e.current_node for i,e in enumerate(self.envs))
        if self.key!=self.signature() or not expected or step==0 and t>0:
            self.replan();t=0
        e=self.envs[index];destination=at(self.paths[index],t+1)
        if destination==e.current_node:
            return -1
        action=e.neighbors_map[e.current_node].index(destination)
        if obs['mask'][action]!=1:
            self.mask_violations+=1
            return -1
        return action


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=[];start=time.perf_counter()
    for robots in (1,3,5,10):
        for path in protocol_paths(a.graph.resolve(),robots):
            protocol=load_protocol(path);regime=protocol.get('regime','static')
            env,_,_=factory(a.graph,robots,seed=33003)
            planner=PrioritizedPlanner(env)
            # A fresh planner is required at every reset, including identical goals.
            original_reset=env.reset
            def reset(*args,**kwargs):
                planner.paths=None;planner.key=None
                return original_reset(*args,**kwargs)
            env.reset=reset
            fns=[lambda obs,i=i:planner.action(i,obs) for i in range(robots)]
            rows=evaluate(env,fns,protocol)
            folder=a.output/f'robots_{robots}'
            write_csv(folder/f'{regime}_detailed.csv',rows)
            summary=summarize(rows)
            summary.update(robots=robots,regime=regime,replans=planner.replans,
                failed_plans=planner.failed_plans,mask_violations=planner.mask_violations)
            write_csv(folder/f'{regime}_summary.csv',[summary]);report.append(summary)
            print(json.dumps(summary),flush=True)
    result={'seconds':time.perf_counter()-start,'results':report}
    assert all(r['collisions_mean']==0 and r['mask_violations']==0 for r in report), 'Planner execution violated reservations'
    (a.output/'complete.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
