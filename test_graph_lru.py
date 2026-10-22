#!/usr/bin/env python3
"""Compile the exact integer Metal kernel body as C++; no GPU/model required.

This checks planner semantics, not Metal compilation or GPU runtime behavior.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]) / 'macos/crates/engine-native/src'
source = (root / 'graph_lru_kernel.hpp').read_text()
body = re.search(r'R"lru\((.*?)\)lru"', source, re.S).group(1)
prelude = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "joint_budget.hpp"
void plan(int E,int K,int S,const int32_t *state,const int32_t *ids,
          int32_t *next_state,int32_t *slots,int32_t *mailbox) {
    struct { unsigned x=0; } thread_position_in_grid;
'''
tests = r'''
}
struct Output { std::vector<int32_t> state,slots,mail; };
Output run(int E,int S,const std::vector<int32_t>&state,const std::vector<int32_t>&ids) {
    int K=ids.size();
    Output o{std::vector<int32_t>(2*S),std::vector<int32_t>(K),std::vector<int32_t>(2+2*K)};
    plan(E,K,S,state.data(),ids.data(),o.state.data(),o.slots.data(),o.mail.data());
    return o;
}
std::vector<int32_t> initial(int S) {
    std::vector<int32_t> st(2*S,-1);
    std::iota(st.begin()+S,st.end(),0);
    return st;
}
int main() {
    // Partial hit later in routing order must survive an earlier miss.
    auto a=run(8,2,initial(2),{0,1});
    auto b=run(8,2,a.state,{2,0});
    assert(b.mail[0]==0 && b.mail[1]==1);
    assert(b.state[b.slots[0]]==2 && b.state[b.slots[1]]==0);
    // Duplicate routes share a slot and load exactly once.
    a=run(8,2,initial(2),{3,3});
    assert(a.mail[1]==1 && a.slots[0]==a.slots[1]);
    auto hit=run(8,2,a.state,{3,3});
    assert(hit.mail[0]==0 && hit.mail[1]==0);
    // Invalid IDs, corrupt owners/permutation, insufficient capacity fail closed.
    assert(run(8,2,initial(2),{-1,0}).mail[0]==2);
    assert(run(8,2,initial(2),{8,0}).mail[0]==2);
    auto bad=initial(2); bad[2]=2;
    assert(run(8,2,bad,{1,2}).mail[0]==1);
    bad=initial(2); bad[3]=bad[2];
    assert(run(8,2,bad,{1,2}).mail[0]==1);
    bad=initial(2); bad[0]=bad[1]=4;
    assert(run(8,2,bad,{1,2}).mail[0]==1);
    assert(run(8,2,initial(2),{1,2,3}).mail[0]==3);
    std::mt19937 rng(12345);
    uint64_t checked=0;
    for (int S: {2,4,8,16,32}) for (int K: {1,2,4}) {
        if(K>S) continue;
        const int E=64;
        auto state=initial(S);
        std::list<int> lru; // independent expert-level oracle
        std::map<int,int> payload; // slot -> expert marker; simulate SSD fills
        for (int step=0; step<3000; ++step) {
            std::vector<int32_t> ids(K);
            for (auto &id:ids) id=(step%5 ? rng()%12 : rng()%E);
            const auto old=state;
            auto out=run(E,S,state,ids);
            assert(state==old); // planner never changes published input state
            assert(out.mail[0]==0);
            int misses=0;
            for (int id:ids) {
                auto it=std::find(lru.begin(),lru.end(),id);
                if (it==lru.end()) {
                    ++misses;
                    if (int(lru.size())==S) {
                        auto victim=lru.end();
                        do { --victim; } while(std::find(ids.begin(),ids.end(),*victim)!=ids.end());
                        lru.erase(victim);
                    }
                } else lru.erase(it);
                lru.push_front(id);
            }
            assert(out.mail[1]==misses);
            std::set<int> destinations;
            for (int m=0;m<misses;++m) {
                int dest=out.mail[2+K+m];
                assert(dest>=0 && dest<S && destinations.insert(dest).second);
                payload[dest]=out.mail[2+m];
            }
            for(int j=0;j<K;++j) {
                assert(out.slots[j]>=0 && out.slots[j]<S);
                assert(payload.at(out.slots[j])==ids[j]);
            }
            std::vector<int> order;
            for(int i=0;i<S;++i) if(out.state[out.state[S+i]]>=0) order.push_back(out.state[out.state[S+i]]);
            assert(order==std::vector<int>(lru.begin(),lru.end()));
            state=out.state; ++checked;
        }
    }
    // Two generations + transient fill can coexist, but not bypass the cap.
    auto budget=std::make_shared<e0n::JointBudget>(100);
    auto old=budget->try_acquire(40), replacement=budget->try_acquire(40);
    auto queued_reader=old;
    old.reset();
    assert(budget->metrics().live==80 && !budget->try_acquire(30));
    queued_reader.reset();
    assert(budget->metrics().live==40 && budget->try_acquire(30));
    replacement.reset(); assert(budget->metrics().live==0);
    std::cout << checked << " graph LRU oracle steps; protected hits, duplicate routes, invalid state, fill mapping and lease lifetime checks passed\n";
}
'''
with tempfile.TemporaryDirectory(prefix='edge0-graph-lru-') as folder:
    cpp = Path(folder) / 'test.cpp'
    exe = Path(folder) / 'test'
    cpp.write_text(prelude + body + tests)
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++20', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-I'+str(root), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
