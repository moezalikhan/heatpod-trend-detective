import numpy as np

from pipeline.heatpod.wetbulb import wet_bulb_temperature


def test_matches_published_example():
    # Stull (2011): 20 C and 50 % humidity gives about 13.7 C
    assert abs(wet_bulb_temperature(20.0, 50.0) - 13.7) < 0.1


def test_wet_bulb_is_below_air_temperature():
    for t, rh in [(20, 50), (30, 80), (35, 40), (40, 20)]:
        assert wet_bulb_temperature(t, rh) < t


def test_wetter_air_gives_higher_wet_bulb():
    assert wet_bulb_temperature(30.0, 80.0) > wet_bulb_temperature(30.0, 30.0)


def test_outside_valid_range_is_nan():
    assert np.isnan(wet_bulb_temperature(30.0, 2.0))
    assert np.isnan(wet_bulb_temperature(60.0, 50.0))


def test_works_on_arrays():
    out = wet_bulb_temperature(np.array([20.0, 30.0]), np.array([50.0, 80.0]))
    assert out.shape == (2,)
