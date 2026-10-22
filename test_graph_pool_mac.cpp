// Tiny Metal runtime check before any model loading. Link to the built libmlx.
#include <iostream>
#include <stdexcept>
#include <vector>
#include <mlx/mlx.h>
#include "graph_lru_kernel.hpp"
namespace mx=mlx::core;
static void require(bool ok) { if(!ok) throw std::runtime_error("Metal graph pool check failed"); }
int main() {
  try {
    auto kernel=mx::fast::metal_kernel("edge0_graph_lru_probe",{"state","ids"},
        {"next_state","slots","mailbox"},e0n::graph_lru_source);
    auto plan=[&](const mx::array& state,const mx::array& ids) {
      return kernel({state,ids},{{4},{2},{6}}, {mx::int32,mx::int32,mx::int32},
          {1,1,1},{1,1,1},{{"E",8},{"K",2},{"S",2}},std::nullopt,false,mx::Device::gpu);
    };
    auto a=plan(mx::array({-1,-1,0,1},{4},mx::int32),mx::array({0,1},{2},mx::int32));
    mx::eval(a);
    require(a[2].data<int32_t>()[0]==0 && a[2].data<int32_t>()[1]==2);
    auto old=mx::array({10.f,20.f},{2,1,1});
    auto pending_old=mx::take(old,a[1],0);
    auto b=plan(a[0],mx::array({2,0},{2},mx::int32));
    mx::eval(b);
    const auto *mail=b[2].data<int32_t>();
    require(mail[0]==0 && mail[1]==1 && mail[2]==2 && mail[4]==1);
    auto filled=mx::scatter(old,{mx::array({mail[4]},{1},mx::int32)},
                            mx::array({30.f},{1,1,1,1}),{0});
    auto routed=mx::take(filled,b[1],0);
    mx::eval(routed,pending_old);
    require(routed.data<float>()[0]==30.f && routed.data<float>()[1]==10.f);
    require(pending_old.data<float>()[0]==10.f && pending_old.data<float>()[1]==20.f);
    auto hit=plan(b[0],mx::array({2,0},{2},mx::int32));
    mx::eval(hit);
    require(hit[2].data<int32_t>()[0]==0 && hit[2].data<int32_t>()[1]==0);
    auto bad=plan(hit[0],mx::array({8,0},{2},mx::int32));
    mx::eval(bad); require(bad[2].data<int32_t>()[0]==2);
    std::cout << "Metal planning, protected hit, fill/gather, immutable old reader, all-hit and invalid-ID checks passed\n";
    return 0;
  } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
