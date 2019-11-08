NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT void Scene<Float, Spectrum>::accel_init_cpu(const Properties &props) {
    ShapeKDTree *kdtree = new ShapeKDTree(props);
    kdtree->inc_ref();
    m_accel = kdtree;
}

MTS_VARIANT void Scene<Float, Spectrum>::accel_release_cpu() {
    ((ShapeKDTree *) m_accel)->dec_ref();
    m_accel = nullptr;
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_cpu(const Ray3f &ray, Mask active) const {
    ScopedPhase sp(ProfilerPhase::RayIntersect);
    const ShapeKDTree *kdtree = (const ShapeKDTree *) m_accel;
    Float cache[MTS_KD_INTERSECTION_CACHE_SIZE];

    auto [hit, hit_t] = kdtree->ray_intersect<false>(ray, cache, active);

    SurfaceInteraction3f si;
    if (likely(any(hit))) {
        ScopedPhase sp2(ProfilerPhase::CreateSurfaceInteraction);
        si = kdtree->create_surface_interaction(ray, hit_t, cache, hit);
    } else {
        si.wavelengths = ray.wavelength;
        si.wi = -ray.d;
    }

    return si;
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_naive_cpu(const Ray3f &ray, Mask active) const {
    ScopedPhase sp(ProfilerPhase::RayIntersect);
    const ShapeKDTree *kdtree = (const ShapeKDTree *) m_accel;
    Float cache[MTS_KD_INTERSECTION_CACHE_SIZE];
    auto [hit, hit_t] = kdtree->ray_intersect_naive<false>(ray, cache, active);

    SurfaceInteraction3f si;
    if (likely(any(hit))) {
        ScopedPhase sp2(ProfilerPhase::CreateSurfaceInteraction);
        si = kdtree->create_surface_interaction(ray, hit_t, cache, hit);
    }

    return si;
}

MTS_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test_cpu(const Ray3f &ray, Mask active) const {
    ScopedPhase p(ProfilerPhase::RayTest);
    const ShapeKDTree *kdtree = (ShapeKDTree *) m_accel;
    return kdtree->ray_intersect<true>(ray, (Float *) nullptr, active).first;
}

NAMESPACE_END(mitsuba)
