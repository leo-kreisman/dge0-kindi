#include "joint_budget.hpp"
#include "prefixcache.hpp"
#include "genstream.hpp"
#include <cassert>
#include <atomic>
#include <iostream>
#include <thread>

// Stand-ins only for sampling and the unused prefetch planner. StreamBase and
// PrefixCache themselves are the real upstream/modified classes under test.
namespace e0n {
Step live_next_step(const PrerouterState &) { return {}; }
namespace sampling {
int32_t sample_step(const std::vector<float> &v, const Params &, bool,
                   const std::vector<int32_t> &, Rng &) {
  return int32_t(std::max_element(v.begin(), v.end()) - v.begin());
}
}}

class CountingStream : public e0n::gen::StreamBase {
 public:
  std::vector<int32_t> evaluated;
  CountingStream(e0n::sampling::Params p) : StreamBase({10,11}, p) {}
 protected:
  bool prefill(const std::vector<int32_t> &t, std::vector<float> &lg,
               std::vector<e0n::RouteRec> &, std::string &) override {
    evaluated = t; lg = {0,1}; return true;
  }
  bool step(int32_t t, std::vector<float> &lg,
            std::vector<e0n::RouteRec> &, std::string &) override {
    evaluated.push_back(t); lg = {0,1}; return true;
  }
};

int main() {
  using namespace e0n;
  {
    JointBudget b(100);
    auto a=b.acquire(80); assert(a);
    assert(!b.acquire(21));
    auto retained=a; a.reset(); assert(b.metrics().live==80);
    retained.reset(); assert(b.metrics().live==0);
    assert(!b.acquire(UINT64_MAX));
    std::atomic<int> successes{0};
    std::vector<std::thread> workers;
    for(int i=0;i<4;++i) workers.emplace_back([&]{
      for(int j=0;j<1000;++j) if(auto lease=b.acquire(40)) {
        ++successes; assert(b.metrics().live<=100);
      }
    });
    for(auto &t:workers)t.join();
    assert(successes>0 && b.metrics().live==0 && b.metrics().peak<=100);
  }
  {
    PrefixCache c({true,100,1});
    auto v=std::make_shared<int>(7);
    assert(c.put("model",{1},v,40));
    assert(c.put("model",{2},v,40));
    assert(c.put("model",{1},v,90)); // replacement used to exceed the budget
    assert(c.metrics().bytes==90 && c.metrics().entries==1);
    assert(!c.lookup("model",{2}).snapshot);
    assert(!c.lookup("different-model",{1}).snapshot);
    assert(!c.lookup("model",{3}).snapshot);
    assert(c.lookup("model",{1,4}).matched_tokens==1);
    assert(!c.put("model",{4},v,101));
    assert(!c.put("model",{4},{},1));
    assert(c.put("model",{5,6},v,90,nullptr,false));
    assert(!c.lookup("model",{5,6}).snapshot); // intermediate state has no logits
    assert(c.lookup("model",{5,6,7}).matched_tokens==2);
    assert(c.evict_one() && !c.evict_one());
    PrefixCache off({false,100,1}); assert(!off.put("m",{1},v,1));
  }
  {
    JointBudget b(100);
    PrefixCache c({true,100,1});
    struct Payload { std::shared_ptr<JointBudget::Lease> lease; };
    auto p=std::make_shared<Payload>(); p->lease=b.acquire(80);
    c.put("m",{1},p,80); p.reset();
    b.add_evictor([&]{return c.evict_one();});
    auto held=c.lookup("m",{1}).snapshot;
    assert(!b.acquire(30)); // eviction must NOT free the live consumer's lease
    held.reset();
    auto expert=b.acquire(100); assert(expert && b.metrics().live==100);
  }
  for(size_t cap: {size_t(0),size_t(1),size_t(3)}) {
    sampling::Params p; p.max_new_tokens=cap;
    CountingStream s(p); bool saved=false;
    s.set_on_complete([&](const auto &ids,const auto &lg){
      saved=true; assert(ids==s.evaluated); assert(!lg.empty());
    });
    int32_t tok; while(s.next_token(tok)) {}
    assert(saved==(cap>0));
    if(cap>0) assert(s.evaluated.size()==2+cap-1);
  }
  {
    sampling::Params p; p.max_new_tokens=10; p.stop_seqs={{1,1}};
    CountingStream s(p); bool saved=false;
    s.set_on_complete([&](const auto &ids,const auto &){
      saved=true; assert(ids==s.evaluated); assert(ids.size()==3);
    });
    int32_t tok; while(s.next_token(tok)) {}
    assert(saved && s.history().size()==2);
  }
  {
    sampling::Params p; p.max_new_tokens=4;
    CountingStream s(p); int checkpoints=0;
    s.set_on_checkpoint(2,[&](const auto &ids,const auto &){
      ++checkpoints; assert(ids==s.evaluated && ids.size()==4);
    });
    int32_t tok; while(s.next_token(tok)) {}
    assert(checkpoints==1);
  }
  {
    sampling::Params p; CountingStream s(p); bool saved=false;
    s.set_on_complete([&](const auto &,const auto &){saved=true;});
    s.cancel(); int32_t t; assert(!s.next_token(t) && !saved);
  }
  {
    sampling::Params p; p.eos_ids={1};
    CountingStream s(p); bool saved=false;
    s.set_on_complete([&](const auto &ids,const auto &){
      saved=true; assert(ids==s.evaluated && ids.size()==2);
    });
    int32_t t; assert(!s.next_token(t)); assert(saved && s.generated()==0);
  }
  {
    sampling::Params p; p.max_new_tokens=1;
    CountingStream s(p); s.evaluated={10,11};
    s.set_prefill_prompt({}); s.set_cached_logits({0,1});
    s.set_on_prefill([](const auto &,const auto &){assert(false);});
    bool saved=false;
    s.set_on_complete([&](const auto &ids,const auto &){
      saved=true; assert(ids==s.evaluated);
    });
    int32_t t; assert(s.next_token(t) && t==1); assert(!s.next_token(t)); assert(saved);
  }
  std::cout << "PASS: joint admission, concurrency, pinned leases, replacement eviction, identity,\n"
               "prefix-only checkpoints, token caps, stop strings, EOS, cached logits, cancellation, checkpoint boundaries\n";
}
