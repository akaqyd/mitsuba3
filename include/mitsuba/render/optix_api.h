#pragma once

#if defined(MI_ENABLE_CUDA)

#include <iomanip>
#include <mitsuba/core/platform.h>

// =====================================================
//       Various opaque handles and enumerations
// =====================================================

using CUdeviceptr            = void*;
using CUstream               = void*;
using OptixPipeline          = void *;
using OptixModule            = void *;
using OptixProgramGroup      = void *;
using OptixResult            = int;
using OptixTraversableHandle = unsigned long long;
using OptixBuildOperation    = int;
using OptixBuildInputType    = int;
using OptixVertexFormat      = int;
using OptixIndicesFormat     = int;
using OptixTransformFormat   = int;
using OptixAccelPropertyType = int;
using OptixProgramGroupKind  = int;
using OptixDeviceContext     = void*;
using OptixTask              = void*;

// =====================================================
//            Commonly used OptiX constants
// =====================================================

#define OPTIX_BUILD_INPUT_TYPE_TRIANGLES         0x2141
#define OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES 0x2142
#define OPTIX_BUILD_INPUT_TYPE_INSTANCES         0x2143
#define OPTIX_BUILD_OPERATION_BUILD              0x2161
// Qiyuan: SP curve
#define OPTIX_BUILD_INPUT_TYPE_CURVES            0x2145


#define OPTIX_GEOMETRY_FLAG_NONE           0
#define OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT 1

#define OPTIX_INDICES_FORMAT_UNSIGNED_INT3 0x2103
#define OPTIX_VERTEX_FORMAT_FLOAT3         0x2121
#define OPTIX_SBT_RECORD_ALIGNMENT         16ull
#define OPTIX_SBT_RECORD_HEADER_SIZE       32

#define OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0
#define OPTIX_COMPILE_OPTIMIZATION_DEFAULT       0
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0       0x2340
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE           0x2350
#define OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL        0x2351

// for curves
#define OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS 1u << 4

#define OPTIX_BUILD_FLAG_ALLOW_COMPACTION  2
#define OPTIX_BUILD_FLAG_PREFER_FAST_TRACE 4
#define OPTIX_PROPERTY_TYPE_COMPACTED_SIZE 0x2181

#define OPTIX_EXCEPTION_FLAG_NONE           0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW 1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH    2
#define OPTIX_EXCEPTION_FLAG_USER           4
#define OPTIX_EXCEPTION_FLAG_DEBUG          8

#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY 0
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING (1u << 1)

#define OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM   (1 << 0)
#define OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE (1 << 31)
// for curves
#define OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE (1 << 2)
#define OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR        (1 << 3)

#define OPTIX_PROGRAM_GROUP_KIND_MISS      0x2422
#define OPTIX_PROGRAM_GROUP_KIND_HITGROUP  0x2424

#define OPTIX_INSTANCE_FLAG_NONE              0
#define OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM (1u << 6)

#define OPTIX_RAY_FLAG_NONE                   0
#define OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT (1u << 2)
#define OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT     (1u << 3)

#define OPTIX_MODULE_COMPILE_STATE_COMPLETED 0x2364

// =====================================================
//          Commonly used OptiX data structures
// =====================================================

struct OptixMotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct OptixAccelBuildOptions {
    unsigned int buildFlags;
    OptixBuildOperation operation;
    OptixMotionOptions motionOptions;
};

struct OptixAccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

struct OptixBuildInputTriangleArray {
    const CUdeviceptr* vertexBuffers;
    unsigned int numVertices;
    OptixVertexFormat vertexFormat;
    unsigned int vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int numIndexTriplets;
    OptixIndicesFormat indexFormat;
    unsigned int indexStrideInBytes;
    CUdeviceptr preTransform;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
    OptixTransformFormat transformFormat;
};

struct OptixBuildInputCustomPrimitiveArray {
    const CUdeviceptr* aabbBuffers;
    unsigned int numPrimitives;
    unsigned int strideInBytes;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
};

struct OptixBuildInputInstanceArray {
    CUdeviceptr instances;
    unsigned int numInstances;
};

typedef enum OptixPrimitiveType
{
    /// Custom primitive.
    OPTIX_PRIMITIVE_TYPE_CUSTOM                        = 0x2500,
    /// B-spline curve of degree 2 with circular cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE       = 0x2501,
    /// B-spline curve of degree 3 with circular cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE           = 0x2502,
    /// Piecewise linear curve with circular cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR                  = 0x2503,
    /// Triangle.
    OPTIX_PRIMITIVE_TYPE_TRIANGLE                      = 0x2531,
} OptixPrimitiveType;

/// Curve end cap types, for non-linear curves
///
typedef enum OptixCurveEndcapFlags
{
    /// Default end caps. Round end caps for linear, no end caps for quadratic/cubic.
    OPTIX_CURVE_ENDCAP_DEFAULT                        = 0,
    /// Flat end caps at both ends of quadratic/cubic curve segments. Not valid for linear.
    OPTIX_CURVE_ENDCAP_ON                             = 1 << 0,
} OptixCurveEndcapFlags;


typedef struct OptixBuildInputCurveArray
{
    /// Curve degree and basis
    /// \see #OptixPrimitiveType
    OptixPrimitiveType curveType;
    /// Number of primitives. Each primitive is a polynomial curve segment.
    unsigned int numPrimitives;

    /// Pointer to host array of device pointers, one per motion step. Host array size must match number of
    /// motion keys as set in #OptixMotionOptions (or an array of size 1 if OptixMotionOptions::numKeys is set
    /// to 1). Each per-motion-key device pointer must point to an array of floats (the vertices of the
    /// curves).
    const CUdeviceptr* vertexBuffers;
    /// Number of vertices in each buffer in vertexBuffers.
    unsigned int numVertices;
    /// Stride between vertices. If set to zero, vertices are assumed to be tightly
    /// packed and stride is sizeof( float3 ).
    unsigned int vertexStrideInBytes;

    /// Parallel to vertexBuffers: a device pointer per motion step, each with numVertices float values,
    /// specifying the curve width (radius) corresponding to each vertex.
    const CUdeviceptr* widthBuffers;
    /// Stride between widths. If set to zero, widths are assumed to be tightly
    /// packed and stride is sizeof( float ).
    unsigned int widthStrideInBytes;

    /// Reserved for future use.
    const CUdeviceptr* normalBuffers;
    /// Reserved for future use.
    unsigned int normalStrideInBytes;

    /// Device pointer to array of unsigned ints, one per curve segment.
    /// This buffer is required (unlike for OptixBuildInputTriangleArray).
    /// Each index is the start of degree+1 consecutive vertices in vertexBuffers,
    /// and corresponding widths in widthBuffers and normals in normalBuffers.
    /// These define a single segment. Size of array is numPrimitives.
    CUdeviceptr indexBuffer;
    /// Stride between indices. If set to zero, indices are assumed to be tightly
    /// packed and stride is sizeof( unsigned int ).
    unsigned int indexStrideInBytes;

    /// Combination of OptixGeometryFlags describing the
    /// primitive behavior.
    unsigned int flag;

    /// Primitive index bias, applied in optixGetPrimitiveIndex().
    /// Sum of primitiveIndexOffset and number of primitives must not overflow 32bits.
    unsigned int primitiveIndexOffset;

    unsigned int endcapFlags;

} OptixBuildInputCurveArray;


struct OptixBuildInput {
    OptixBuildInputType type;
    union {
        OptixBuildInputTriangleArray triangleArray;
        OptixBuildInputCustomPrimitiveArray customPrimitiveArray;
        OptixBuildInputInstanceArray instanceArray;
        /// Qiyuan: Curve inputs.
        OptixBuildInputCurveArray curveArray;
        char pad[1024];
    };
};

struct OptixInstance {
    float transform[12];
    unsigned int instanceId;
    unsigned int sbtOffset;
    unsigned int visibilityMask;
    unsigned int flags;
    OptixTraversableHandle traversableHandle;
    unsigned int pad[2];
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    OptixPayloadType *payloadTypes;
};

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
};

struct OptixAccelEmitDesc {
    CUdeviceptr result;
    OptixAccelPropertyType type;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char* entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixProgramGroupOptions {
    OptixPayloadType *payloadType;
};

struct OptixShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

// =====================================================
//             Commonly used OptiX functions
// =====================================================

#if defined(OPTIX_API_IMPL)
#  define D(name, ...) OptixResult (*name)(__VA_ARGS__) = nullptr;
#else
#  define D(name, ...) extern MI_EXPORT_LIB OptixResult (*name)(__VA_ARGS__)
#endif

D(optixAccelComputeMemoryUsage, OptixDeviceContext,
  const OptixAccelBuildOptions *, const OptixBuildInput *, unsigned int,
  OptixAccelBufferSizes *);
D(optixAccelBuild, OptixDeviceContext, CUstream, const OptixAccelBuildOptions *,
  const OptixBuildInput *, unsigned int, CUdeviceptr, size_t, CUdeviceptr,
  size_t, OptixTraversableHandle *, const OptixAccelEmitDesc *, unsigned int);
D(optixModuleCreateFromPTXWithTasks, OptixDeviceContext,
  const OptixModuleCompileOptions *, const OptixPipelineCompileOptions *,
  const char *, size_t, char *, size_t *, OptixModule *, OptixTask *);
D(optixModuleGetCompilationState, OptixModule, int *);
D(optixModuleDestroy, OptixModule);
D(optixTaskExecute, OptixTask, OptixTask *, unsigned int, unsigned int *);
D(optixProgramGroupCreate, OptixDeviceContext, const OptixProgramGroupDesc *,
  unsigned int, const OptixProgramGroupOptions *, char *, size_t *,
  OptixProgramGroup *);
D(optixProgramGroupDestroy, OptixProgramGroup);
D(optixSbtRecordPackHeader, OptixProgramGroup, void *);
D(optixAccelCompact, OptixDeviceContext, CUstream, OptixTraversableHandle,
  CUdeviceptr, size_t, OptixTraversableHandle *);

#undef D

NAMESPACE_BEGIN(mitsuba)
extern MI_EXPORT_LIB void optix_initialize();
extern MI_EXPORT_LIB void optix_shutdown();
NAMESPACE_END(mitsuba)

#endif // defined(MI_ENABLE_CUDA)
