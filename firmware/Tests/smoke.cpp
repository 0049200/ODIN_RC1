#include "odin_native.hpp"
#include <cstdio>
#include <cassert>
static odin::native::Engine e;
static odin::native::Output o;
int main(){e.init();e.d.set_class_model(0,0,.1f);e.d.set_class_model(1,1,.1f);e.d.set_class_model(2,2,.1f);unsigned ok=0;for(unsigned i=1;i<=100;i++){odin::native::Input in{};in.sensors.frame_id=i;in.sensors.timestamp_s=i*.05f;in.sensors.valid_mask=255;in.quality=1;in.importance=.5f;in.trusted=1;in.runtime.crc_ok=1;in.runtime.watchdog_fed=1;for(unsigned j=0;j<4;j++){in.sensors.primary[j]=in.sensors.redundant[j]=.1f;in.relevance[j]=128;}in.raw_discrimination=.1f;if(e.tick(in,o))++ok;assert(!o.ka.actuation_enabled);assert(!o.x.actuator_permitted);}printf("ticks=%u engine=%zu output=%zu memory=%zu KA=%zu\n",ok,sizeof(e),sizeof(o),sizeof(e.m),sizeof(e.ka));assert(ok==100);assert(e.m.verify_integrity());assert(e.p.samples_seen()==100);assert(e.rps.ticks_processed()==100);assert(e.s.frames_processed()==100);assert(o.x.sequence==100);}
