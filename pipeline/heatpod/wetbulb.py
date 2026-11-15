"""Wet-bulb temperature from air temperature and relative humidity.

Uses the Stull (2011) approximation, which needs only the two variables the
pipeline already has: T2M (air temperature, degrees C) and RH2M (relative
humidity, percent). Valid for roughly RH 5 to 99 % and T -20 to 50 C; outside
that range the result is returned as NaN instead of a misleading number.

Reference: Stull, R. (2011). Wet-bulb temperature from relative humidity and
air temperature. Journal of Applied Meteorology and Climatology, 50, 2267-2269.
"""

import numpy as np


def wet_bulb_temperature(t2m_c, rh2m_pct):
    """Return wet-bulb temperature in degrees C (NaN outside the valid range)."""
    t = np.asarray(t2m_c, dtype=float)
    rh = np.asarray(rh2m_pct, dtype=float)
    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    valid = (rh >= 5) & (rh <= 99) & (t >= -20) & (t <= 50)
    return np.where(valid, tw, np.nan)
