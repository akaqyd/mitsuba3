#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/mmap.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#if defined(MI_ENABLE_EMBREE)
#include <embree3/rtcore.h>
#endif

#if defined(MI_ENABLE_CUDA)
    #include "optix/bspline.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-bspline:

BSpline(:monosp:`bspline`)
-------------------------------------------------

.. pluginparameters::

 * - file_path
   - |string|
   - The path to the file that contains all the control points for the B-spline

 * - to_world
   - |transform|
   -  Specifies an optional linear object-to-world transformation.
      Note that non-uniform scales and shears are not permitted!
      (Default: none, i.e. object space = world space)

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_basic.jpg
   :caption: Basic example
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_parameterization.jpg
   :caption: A textured sphere with the default parameterization
.. subfigend::
   :label: fig-sphere

This shape plugin describes a simple sphere intersection primitive. It should
always be preferred over sphere approximations modeled using triangles.

A sphere can either be configured using a linear :monosp:`to_world` transformation or the :monosp:`center` and :monosp:`radius` parameters (or both).
The two declarations below are equivalent.

.. tabs::
    .. code-tab:: xml
        :name: sphere

        <shape type="sphere">
            <transform name="to_world">
                <scale value="2"/>
                <translate x="1" y="0" z="0"/>
            </transform>
            <bsdf type="diffuse"/>
        </shape>

        <shape type="sphere">
            <point name="center" x="1" y="0" z="0"/>
            <float name="radius" value="2"/>
            <bsdf type="diffuse"/>
        </shape>

    .. code-tab:: python

        'sphere_1': {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f.scale([2, 2, 2]).translate([1, 0, 0]),
            'bsdf': {
                'type': 'diffuse'
            }
        },

        'sphere_2': {
            'type': 'sphere',
            'center': [1, 0, 0],
            'radius': 2,
            'bsdf': {
                'type': 'diffuse'
            }
        }

When a :ref:`sphere <shape-sphere>` shape is turned into an :ref:`area <emitter-area>`
light source, Mitsuba 3 switches to an efficient
`sampling strategy <https://www.akalin.com/sampling-visible-sphere>`_ by Fred Akalin that
has particularly low variance.
This makes it a good default choice for lighting new scenes.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_light_mesh.jpg
   :caption: Spherical area light modeled using triangles
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_light_analytic.jpg
   :caption: Spherical area light modeled using the :ref:`sphere <shape-sphere>` plugin
.. subfigend::
   :label: fig-sphere-light
 */

template <bool Negate, size_t N>
void advance(const char **start_, const char *end, const char (&delim)[N]) {
    const char *start = *start_;

    while (true) {
        bool is_delim = false;
        for (size_t i = 0; i < N; ++i)
            if (*start == delim[i])
                is_delim = true;
        if ((is_delim ^ Negate) || start == end)
            break;
        ++start;
    }

    *start_ = start;
}


template <typename Float, typename Spectrum>
class BSpline final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape, m_to_world, m_to_object, m_is_instance, initialize,
                   mark_dirty, get_children_string, parameters_grad_enabled)
    MI_IMPORT_TYPES()

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;
//    using typename Base::ScalarRay3f;

    using InputFloat = float;
    using InputPoint3f  = Point<InputFloat, 3>;
    using InputVector3f = Vector<InputFloat, 3>;
    using FloatStorage = DynamicBuffer<dr::replace_scalar_t<Float, InputFloat>>;

    using UInt32Storage = DynamicBuffer<UInt32>;


    BSpline(const Properties &props) : Base(props) {

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        std::string m_name = file_path.filename().string();

        // used for throwing an error later
        auto fail = [&](const char *descr, auto... args) {
            Throw(("Error while loading bspline curve file \"%s\": " + std::string(descr))
                      .c_str(), m_name, args...);
        };

        Log(Debug, "Loading a bspline curve file from \"%s\" ..", m_name);
        if (!fs::exists(file_path))
            fail("file not found");

        ref<MemoryMappedFile> mmap = new MemoryMappedFile(file_path);
        ScopedPhase phase(ProfilerPhase::LoadGeometry);

        // temporary buffer for vertices and per-vertex radius
        std::vector<InputVector3f> vertices;
        std::vector<InputFloat> radius;
        size_t vertex_guess = mmap->size() / 100;
        vertices.reserve(vertex_guess);

        // load data from the given .txt file
        const char *ptr = (const char *) mmap->data();
        const char *eof = ptr + mmap->size();
        char buf[1025];
        Timer timer;

        while (ptr < eof) {
            // Determine the offset of the next newline
            const char *next = ptr;
            advance<false>(&next, eof, "\n");

            // Copy buf into a 0-terminated buffer
            size_t size = next - ptr;
            if (size >= sizeof(buf) - 1)
                fail("file contains an excessively long line! (%i characters)", size);
            memcpy(buf, ptr, size);
            buf[size] = '\0';

            // handle current line: v.x v.y v.z radius
            // skip whitespace
            const char *cur = buf, *eol = buf + size;
            advance<true>(&cur, eol, " \t\r");
            bool parse_error = false;

            // Vertex position
            InputPoint3f p;
            InputFloat r;
            for (size_t i = 0; i < 3; ++i) {
                const char *orig = cur;
                p[i] = string::strtof<InputFloat>(cur, (char **) &cur);
                parse_error |= cur == orig;
            }

            // parse per-vertex radius
            const char *orig = cur;
            r = string::strtof<InputFloat>(cur, (char **) &cur);
            parse_error |= cur == orig;

            p = m_to_world.scalar().transform_affine(p);

            if (unlikely(!all(dr::isfinite(p))))
                fail("bspline control point contains invalid vertex position data");
            if (unlikely(!dr::isfinite(r)))
                fail("bspline control point contains invalid radius data");

            // TODO: how to calculate bspline's bbox
            // just expand using control points for now
            m_bbox.expand(p);

            vertices.push_back(p);
            radius.push_back(r);

            if (unlikely(parse_error))
                fail("could not parse line \"%s\"", buf);
            ptr = next + 1;
        }

        m_control_point_count = vertices.size();
        m_segment_count = m_control_point_count - 3;
        if (unlikely(m_control_point_count < 4))
            fail("bspline must have at least four control points");


        for (int i = 0; i < m_control_point_count; i++)
            Log(Debug, "Loaded a control point %s with radius %f",
                vertices[i], radius[i]);

        // store the data from the previous temporary buffer
        std::unique_ptr<float[]> vertex_positions_radius(new float[m_control_point_count * 4]);
        std::unique_ptr<ScalarIndex[]> indices(new ScalarIndex[m_segment_count]);

        // for OptiX
        std::unique_ptr<float[]> vertex_position(new float[m_control_point_count * 3]);
        std::unique_ptr<float[]> vertex_radius(new float[m_control_point_count * 1]);

        for (uint i = 0; i < vertices.size(); i++) {
            InputFloat* position_ptr = vertex_positions_radius.get() + i * 4;
            InputFloat* radius_ptr   = vertex_positions_radius.get() + i * 4 + 3;

            dr::store(position_ptr, vertices[i]);
            dr::store(radius_ptr, radius[i]);

            // OptiX
            position_ptr = vertex_position.get() + i * 3;
            radius_ptr = vertex_radius.get() + i;
            dr::store(position_ptr, vertices[i]);
            dr::store(radius_ptr, radius[i]);
        }

        for (uint i = 0; i < m_segment_count; i++) {
            u_int32_t* index_ptr = indices.get() + i;
            dr::store(index_ptr, i);
        }

        m_vertex_with_radius = dr::load<FloatStorage>(vertex_positions_radius.get(), m_control_point_count * 4);
        m_indices = dr::load<UInt32Storage>(indices.get(), m_segment_count);

        // OptiX
        m_vertex = dr::load<FloatStorage>(vertex_position.get(), m_control_point_count * 3);
        m_radius = dr::load<FloatStorage>(vertex_radius.get(), m_control_point_count * 1);

        size_t vertex_data_bytes = 8 * sizeof(InputFloat);
        Log(Debug, "\"%s\": read %i control points (%s in %s)",
            m_name, m_control_point_count,
            util::mem_string(m_control_point_count * vertex_data_bytes),
            util::time_string((float) timer.value())
        );

        update();
        initialize();
    }

    void update() {
        // TODO
    }

    bool is_curve() const override {
        return true;
    }

    bool is_bspline_curve() const override {
        return true;
    }

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    template <typename FloatP, typename Ray3fP>
    std::tuple<FloatP, Point<FloatP, 2>, dr::uint32_array_t<FloatP>,
               dr::uint32_array_t<FloatP>>
    ray_intersect_preliminary_impl(const Ray3fP &ray,
                                   dr::mask_t<FloatP> active) const {
        NotImplementedError("ray_intersect_preliminary_impl");
    }

    template <typename FloatP, typename Ray3fP>
    dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray,
                                     dr::mask_t<FloatP> active) const {
        NotImplementedError("ray_test_impl");
    }

    MI_SHAPE_DEFINE_RAY_INTERSECT_METHODS()

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     const PreliminaryIntersection3f &pi,
                                                     uint32_t ray_flags,
                                                     uint32_t recursion_depth,
                                                     Mask active) const override {
        // TODO
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        return si;
    }
    //! @}
    // =============================================================



    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    // Sampling routines are not implemented for B-spline curves

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        NotImplementedError("sample_position");
    }
    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        NotImplementedError("pdf_position");
    }
    DirectionSample3f sample_direction(const Interaction3f &it, const Point2f &sample,
                                       Mask active) const override {
        NotImplementedError("sample_direction");
    }
    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        NotImplementedError("pdf_direction");
    }
    //! @}
    // =============================================================



#if defined(MI_ENABLE_EMBREE)
    RTCGeometry embree_geometry(RTCDevice device) override {
        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE);

        rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4,
                                   m_vertex_with_radius.data(), 0, 4 * sizeof(InputFloat),
                                   m_control_point_count);
        rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT,
                                   m_indices.data(), 0, 1 * sizeof(ScalarIndex),
                                   m_segment_count);
        rtcCommitGeometry(geom);
        return geom;
    }
#endif


#if defined(MI_ENABLE_CUDA)
    void optix_prepare_geometry() override { }

    void optix_build_input(OptixBuildInput &build_input) const override {
        m_vertex_buffer_ptr = (void*) m_vertex.data(); // triggers dr::eval()
        m_radius_buffer_ptr = (void*) m_radius.data(); // triggers dr::eval()
        m_index_buffer_ptr = (void*) m_indices.data(); // triggers dr::eval()

        build_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
        build_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        build_input.curveArray.numPrimitives        = m_segment_count;

        build_input.curveArray.vertexBuffers        = (CUdeviceptr*) &m_vertex_buffer_ptr;
        build_input.curveArray.numVertices          = m_control_point_count;
        build_input.curveArray.vertexStrideInBytes  = sizeof( InputFloat ) * 3;

        build_input.curveArray.widthBuffers         = (CUdeviceptr*) &m_radius_buffer_ptr;
        build_input.curveArray.widthStrideInBytes   = sizeof( InputFloat );

        build_input.curveArray.indexBuffer          = (CUdeviceptr) m_index_buffer_ptr;
        build_input.curveArray.indexStrideInBytes   = sizeof( ScalarIndex );

        build_input.curveArray.normalBuffers        = 0;
        build_input.curveArray.normalStrideInBytes  = 0;
        build_input.curveArray.flag                 = OPTIX_GEOMETRY_FLAG_NONE;
        build_input.curveArray.primitiveIndexOffset = 0;
        build_input.curveArray.endcapFlags          = OptixCurveEndcapFlags::OPTIX_CURVE_ENDCAP_ON;

        Log(Debug, "Optix_build_input done for one BSpline curve, numVertices %d, numPrimitives %d",
                    m_control_point_count, m_segment_count);
    }
#endif


    ScalarBoundingBox3f bbox() const override {
        return m_bbox;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BSpline[" << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    ScalarBoundingBox3f m_bbox;

    ScalarSize m_control_point_count = 0;
    ScalarSize m_segment_count = 0;

    mutable FloatStorage m_vertex_with_radius;
    // mutable DynamicBuffer<ScalarIndex> m_indices;
    mutable UInt32Storage m_indices;

    // separate storage of control points and per-vertex radius for OptiX
    mutable FloatStorage m_vertex;
    mutable FloatStorage m_radius;

    // for OptiX build input
    mutable void* m_vertex_buffer_ptr = nullptr;
    mutable void* m_radius_buffer_ptr = nullptr;
    mutable void* m_index_buffer_ptr = nullptr;
};

MI_IMPLEMENT_CLASS_VARIANT(BSpline, Shape)
MI_EXPORT_PLUGIN(BSpline, "BSpline intersection primitive");
NAMESPACE_END(mitsuba)
