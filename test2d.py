import numpy as np
from halotools.utils import normalized_vectors
from modular_alignment_2d import align_to_axis, align_vector_to_axis, align_angle_to_axis, axes_correlated_with_input_vector, align_to_tidal_field, align_radially, \
                                    align_randomly, tidal_angle

def test_align_vector_to_axis():
    N = 100000
    Av = np.random.uniform(0, 1, (N,2))

    major, minor = align_vector_to_axis(Av)
    assert( (Av == major ).all() )

def test_align_angle_to_axis():
    N = 100000
    Av = np.random.uniform(0, np.pi, N)

    angles = align_angle_to_axis(Av)
    assert( (Av == angles ).all() )

def test_axes_correlated_with_input_vector():
    N = 100000
    Av = normalized_vectors( np.random.uniform(0, 1, (N,2)) )
    angles = np.random.uniform(0, np.pi, N)

    # Test with as_vector = True
    strong_major = axes_correlated_with_input_vector(Av, 1, as_vector=True)
    mid_major = axes_correlated_with_input_vector(Av, 0.5, as_vector=True)
    weak_major = axes_correlated_with_input_vector(Av, 0, as_vector=True)
    # Because vonMises does not yet go to perfectly aligned or anti-aligned, we have to look at more average properties
    strong_diff = np.mean( abs( Av - strong_major ) )
    mid_diff = np.mean( abs( Av - mid_major ) )
    weak_diff = np.mean( abs( Av - weak_major ) )

    assert( strong_diff < mid_diff )
    assert( mid_diff < weak_diff )

    # Test with as_vector = False
    strong_major = axes_correlated_with_input_vector(angles, 1, as_vector=False)
    mid_major = axes_correlated_with_input_vector(angles, 0.5, as_vector=False)
    weak_major = axes_correlated_with_input_vector(angles, 0, as_vector=False)
    # Because vonMises does not yet go to perfectly aligned or anti-aligned, we have to look at more average properties
    strong_diff = np.mean( abs( angles - strong_major ) )
    mid_diff = np.mean( abs( angles - mid_major ) )
    weak_diff = np.mean( abs( angles - weak_major ) )

    assert( strong_diff < mid_diff )
    assert( mid_diff < weak_diff )

def test_align_to_axis():
    N = 100000
    Av = normalized_vectors( np.random.uniform(0, 1, (N,2)) )
    angles = np.random.uniform(0, np.pi, N)

    # Test with as_vector = True
    strong_major = align_to_axis(Av, 1, as_vector=True)
    mid_major = align_to_axis(Av, 0.5, as_vector=True)
    weak_major = align_to_axis(Av, 0, as_vector=True)
    # Because vonMises does not yet go to perfectly aligned or anti-aligned, we have to look at more average properties
    strong_diff = np.mean( abs( Av - strong_major ) )
    mid_diff = np.mean( abs( Av - mid_major ) )
    weak_diff = np.mean( abs( Av - weak_major ) )

    assert( strong_diff < mid_diff )
    assert( mid_diff < weak_diff )

    # Test with as_vector = False
    strong_major = align_to_axis(angles, 1, as_vector=False)
    mid_major = align_to_axis(angles, 0.5, as_vector=False)
    weak_major = align_to_axis(angles, 0, as_vector=False)
    # Because vonMises does not yet go to perfectly aligned or anti-aligned, we have to look at more average properties
    strong_diff = np.mean( abs( angles - strong_major ) )
    mid_diff = np.mean( abs( angles - mid_major ) )
    weak_diff = np.mean( abs( angles - weak_major ) )

    assert( strong_diff < mid_diff )
    assert( mid_diff < weak_diff )

def test_align_to_tidal_field():
    N = 100000
    S = np.random.uniform(-1,1,(N,2,2))
    s11 = S[:,0,0]
    s22 = S[:,1,1]
    s12 = S[:,0,1]
    redshift = np.random.uniform(0,1,N)
    tidal_angles = tidal_angle(s11, s22, s12, redshift)
    tidal_vectors = np.array([np.cos(tidal_angles), np.sin(tidal_angles)]).T

    # Test with as_vector = True
    strong_major = align_to_tidal_field(s11, s22, s12, redshift, 1, as_vector=True)
    mid_major = align_to_tidal_field(s11, s22, s12, redshift, 0.5, as_vector=True)
    weak_major = align_to_tidal_field(s11, s22, s12, redshift, 0, as_vector=True)
    # Because vonMises does not yet go to perfectly aligned or anti-aligned, we have to look at more average properties
    strong_diff = np.mean( abs( tidal_vectors - strong_major ) )
    mid_diff = np.mean( abs( tidal_vectors - mid_major ) )
    weak_diff = np.mean( abs( tidal_vectors - weak_major ) )

    assert( strong_diff < mid_diff )
    assert( mid_diff < weak_diff )

    # Test with as_vector = False
    strong_major = align_to_tidal_field(s11, s22, s12, redshift, 1, as_vector=False)
    mid_major = align_to_tidal_field(s11, s22, s12, redshift, 0.5, as_vector=False)
    weak_major = align_to_tidal_field(s11, s22, s12, redshift, 0, as_vector=False)
    # Because vonMises does not yet go to perfectly aligned or anti-aligned, we have to look at more average properties
    strong_diff = np.mean( abs( tidal_angles - strong_major ) )
    mid_diff = np.mean( abs( tidal_angles - mid_major ) )
    weak_diff = np.mean( abs( tidal_angles - weak_major ) )

    assert( strong_diff < mid_diff )
    assert( mid_diff < weak_diff )

def test_align_radially():
    pass

def test_align_randomly():
    pass

if __name__ == "__main__":
    print("TESTING")

    test_align_vector_to_axis()
    test_align_angle_to_axis()
    test_axes_correlated_with_input_vector()
    test_align_to_axis()
    test_align_to_tidal_field()
    test_align_radially()
    test_align_randomly()

    print("DONE TESTING")