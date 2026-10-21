#include "expert_residency.hpp"
#include "joint_budget.hpp"
#include <cassert>
#include <iostream>
#include <thread>
using namespace e0n;
using Cache=ExpertResidency<int>;
static void put(Cache &c,int k) { c.put(k,std::make_shared<int>(k)); }
static void rejects(const std::function<void()> &f) {bool threw=false;try{f();}catch(const std::exception&){threw=true;}assert(threw);}
static void write_json(const std::string &path,const nlohmann::json &j){std::ofstream(path)<<j;}
int main() {
  char temp[]="/tmp/edge0-residency-XXXXXX"; assert(mkdtemp(temp));
  std::filesystem::path dir(temp); const auto path=(dir/"profile.json").string();
  {
    Cache c(2); put(c,0);put(c,1);assert(c.get(0));put(c,2);
    assert(c.contains(0)&&c.contains(2)&&!c.contains(1));
    auto live=c.pop(0);assert(*live==0&&!c.contains(0));assert(c.evict_one());assert(!c.evict_one());
    assert(!c.metrics().enabled); // unchanged default LRU
  }
  {
    Cache c(2);c.configure(true,{{0,8}},"model-a",path);
    for(int i=0;i<40;++i)c.observe(0,{0,0}); // duplicate route counted once
    assert(c.metrics().observations==40);
    put(c,0);put(c,1);put(c,2);assert(c.contains(0)&&!c.contains(1));
    // A workload shift eventually replaces the former favourite.
    for(int i=0;i<300;++i)c.observe(0,{2});
    assert(c.evict_one());put(c,3);assert(!c.contains(0)&&c.contains(2));
    auto held=c.get(2);c.evict_one();c.evict_one();assert(*held==2);
    c.save(); assert(std::filesystem::exists(path));
    Cache next(1); next.configure(true,{{0,8}},"model-a",path);
    assert(next.metrics().loaded==2); assert(next.size()==0); // no startup loads
    put(next,2);put(next,7);assert(next.contains(2));
    auto original=std::filesystem::last_write_time(path);
    next.save(); assert(std::filesystem::last_write_time(path)==original); // no writes without observations
    rejects([&]{next.observe(0,{8});});rejects([&]{next.observe(4,{0});});
    rejects([&]{Cache x(1);x.configure(true,{{0,8}},"model-b",path);});
    rejects([&]{Cache x(1);x.configure(true,{{0,9}},"model-a",path);});
    rejects([&]{Cache x(1);x.configure(false,{{0,8}},"model-a",path);});
    std::ifstream in(path); auto good=nlohmann::json::parse(in);
    for(int which=0;which<6;++which) {
      auto bad=good;
      if(which==0)bad["version"]=2;
      if(which==1)bad["heat"][0][1]=999;
      if(which==2)bad["heat"][0][2]=-1;
      if(which==3)bad["heat"][0][2]=129;
      if(which==4)bad["heat"].push_back(bad["heat"][0]);
      if(which==5)bad["heat"][0][0]=0.5;
      write_json(path,bad);
      rejects([&]{Cache x(1);x.configure(true,{{0,8}},"model-a",path);});
    }
    std::ofstream(path)<<"{";rejects([&]{Cache x(1);x.configure(true,{{0,8}},"model-a",path);});
    std::ofstream(path)<<std::string(1048577,' ');rejects([&]{Cache x(1);x.configure(true,{{0,8}},"model-a",path);});
    c.save();Cache restored(1);restored.configure(true,{{0,8}},"model-a",path);
    assert(restored.metrics().loaded==2);
    struct stat st{};assert(stat(path.c_str(),&st)==0);assert((st.st_mode&0777)==0600);
  }
  {
    // Layer-local time: intervening work in another layer cannot age this one.
    Cache c(1);c.configure(true,{{0,8},{1,8}},"m","");c.observe(0,{0});
    for(int i=0;i<1000;++i)c.observe(1,{0});
    put(c,0);put(c,1);assert(c.contains(0));
    // Concurrent cache readers, eviction and observation: one mutex for policy/map.
    std::vector<std::thread> ts;
    for(int t=0;t<4;++t)ts.emplace_back([&,t]{for(int i=0;i<500;++i){c.observe(0,{t});put(c,t);c.get(t);c.evict_one();}});
    for(auto &t:ts)t.join();assert(c.size()<=1);
  }
  {
    auto model=dir/"model";std::filesystem::create_directory(model);std::ofstream(model/"config.json")<<"{}";
    auto a=residency_identity(model.string());std::ofstream(model/"adapter.safetensors")<<"adapter";
    assert(a!=residency_identity(model.string()));
  }
  {
    // Eviction doesn't release bytes while a consumer owns a cached allocation.
    struct Payload {std::shared_ptr<JointBudget::Lease> lease;};
    auto budget=std::make_shared<JointBudget>(100);
    ExpertResidency<Payload> c(1);c.configure(true,{{0,2}},"m","");
    auto held=std::make_shared<Payload>();held->lease=budget->acquire(80);
    c.put(0,held);c.evict_one();assert(budget->metrics().live==80);
    held.reset();assert(budget->metrics().live==0);
  }
  {
    // Deterministic locality fixture, not an SSD/model benchmark.
    auto replay=[](bool adaptive){
      Cache c(4);c.configure(adaptive,{{0,64}},"fixture","");int misses=0;
      for(int step=0;step<200;++step) {
        int hot=(step<100)?0:1;
        for(int e: {hot,hot,hot, 2+(step%60), 2+((step+1)%60), 2+((step+2)%60), 2+((step+3)%60), 2+((step+4)%60)}) {
          c.observe(0,{e}); if(!c.get(e)){++misses;put(c,e);}
        }
      }
      return misses;
    };
    int lru=replay(false),adaptive=replay(true);
    std::cout<<"Synthetic locality misses LRU="<<lru<<" adaptive="<<adaptive<<" (not inference speed)\n";
    assert(adaptive<=lru+5);
  }
  std::filesystem::remove_all(dir);
  std::cout<<"Residency/profile correctness checks passed\n";
}
