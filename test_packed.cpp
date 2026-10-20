#include "packed_experts.hpp"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sys/mman.h>

using namespace e0n;
using json=nlohmann::json;

int main(int argc,char **argv) {
  try {
    if(argc!=4) throw std::runtime_error("usage: test_packed MODEL SIDECAR LAYER");
    const std::string model=argv[1], dir=argv[2];
    const int layer=std::stoi(argv[3]);
    std::ifstream in(dir+"/layer-"+std::to_string(layer)+".json");
    const auto j=json::parse(in);
    std::array<PackedExperts::Source,9> sources;
    for(size_t i=0;i<9;++i) {
      const auto &s=j.at("sources").at(i);
      sources[i]={model+"/"+s.at("shard").get<std::string>(),s.at("shard"),
                  s.at("begin"),s.at("end"),j.at("lengths").at(i)};
    }
    PackedExperts packed;
    packed.open(dir,layer,j.at("experts"),sources);
    std::vector<uint64_t> selected={0,uint64_t(j.at("experts").get<int>()-1)};
    for(uint64_t e=1;e<6 && e<j.at("experts").get<uint64_t>();++e) selected.push_back(e);
    json rows=json::array();
    // No page-cache flushing. Alternate order; count actual bytes/calls, not tokens.
    for(int trial=0;trial<4;++trial) for(int arm=0;arm<2;++arm) {
      const bool use_packed=(arm^(trial&1))!=0;
      uint64_t calls=0,bytes=0;
      const auto start=std::chrono::steady_clock::now();
      for(auto e:selected) {
        std::vector<uint8_t> reference(packed.bytes());
        uint64_t offset=0;
        if(use_packed) {
          auto p=packed.read(e,calls);
          std::memcpy(reference.data(),p.data(),packed.bytes());
        } else {
          for(const auto &s:sources) {
            const int fd=::open(s.path.c_str(),O_RDONLY);
            if(fd<0) throw std::runtime_error("source open failed");
            uint64_t done=0;
            while(done<s.length) {
              ++calls;
              const auto n=::pread(fd,reference.data()+offset+done,s.length-done,
                                   s.begin+e*s.length+done);
              if(n<=0) {::close(fd); throw std::runtime_error("source read failed");}
              done+=n;
            }
            ::close(fd); offset+=s.length;
          }
          if(PackedExperts::crc32(reference.data(),reference.size())!=j.at("crc32").at(e))
            throw std::runtime_error("source vs sidecar mismatch");
        }
        bytes+=packed.bytes();
      }
      const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      rows.push_back({{"trial",trial},{"mode",use_packed?"packed_pread_crc":"source_9pread_crc"},
        {"seconds",seconds},{"read_calls",calls},{"bytes",bytes},{"experts",selected.size()}});
    }
    // Verify individual tensor offsets and all bytes, not only checksums.
    for(auto e:selected) {
      uint64_t calls=0, offset=0;
      auto p=packed.read(e,calls);
      for(const auto &s:sources) {
        std::vector<uint8_t> src(s.length);
        const int fd=::open(s.path.c_str(),O_RDONLY);
        if(fd<0) throw std::runtime_error("source open");
        auto n=::pread(fd,src.data(),s.length,s.begin+e*s.length); ::close(fd);
        if(n!=ssize_t(s.length) || std::memcmp(src.data(),p.data()+offset,s.length))
          throw std::runtime_error("byte parity failed");
        offset+=s.length;
      }
    }
    bool rejected=false;
    try {uint64_t calls=0; packed.read(j.at("experts"),calls);} catch(...) {rejected=true;}
    assert(rejected);
    std::cout<<json({{"byte_parity",true},{"rows",rows},
      {"note","I/O microbenchmark; warm OS cache uncontrolled; not inference or a Mac speed prediction"}}).dump(2)<<"\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<"\n"; return 1;}
}
