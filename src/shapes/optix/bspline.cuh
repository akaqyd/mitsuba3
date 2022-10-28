#pragma once

#include <mitsuba/render/optix/common.h>

#ifdef __CUDACC__
extern "C" __global__ void __intersection__bspline() {

}

extern "C" __global__ void __closesthit__bspline() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
    set_preliminary_intersection_to_payload(
        optixGetRayTmax(), Vector2f(), 0, sbt_data->shape_registry_id);
}
#endif