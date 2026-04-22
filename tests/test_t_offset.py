import unittest
import numpy as np

from sfparticles import Particles, SpinParticles, Fields, RadiationReactionType


class TestTOffset(unittest.TestCase):

    def test_t_offset_default(self):
        """Backward compatibility: no t_offset provided defaults to zeros."""
        p = Particles('e', q=-1, m=1, N=10, props=(np.zeros(10),)*6)
        self.assertTrue('t_offset' in p.attrs)
        self.assertEqual(p.t_offset.shape, (10,))
        self.assertTrue(np.all(p.t_offset == 0))

    def test_t_offset_from_props(self):
        """7-element props for Particles includes t_offset."""
        to = np.array([0.0, 1.0, 2.0] + [0.0]*7)
        p = Particles('e', q=-1, m=1, N=10, props=(np.zeros(10),)*6 + (to,))
        self.assertTrue(np.allclose(p.t_offset, to))

    def test_t_offset_spinparticles(self):
        """10-element props for SpinParticles includes t_offset."""
        to = np.array([0.0, 1.0] + [0.0]*8)
        p = SpinParticles('e', q=-1, m=1, N=10, props=(np.zeros(10),)*9 + (to,))
        self.assertTrue(np.allclose(p.t_offset, to))
        # Also verify backward compat with 9 elements
        p2 = SpinParticles('e', q=-1, m=1, N=10, props=(np.zeros(10),)*9)
        self.assertTrue(np.all(p2.t_offset == 0))

    def test_t_offset_field_evaluation(self):
        """Field eval uses t + t_offset for each particle."""
        def field_func(x, y, z, t):
            return (t, 0, 0, 0, 0, 0)
        fields = Fields(field_func)
        to = np.array([0.0, 1.0, 2.0])
        p = Particles('e', q=-1, m=1, N=3, props=(np.zeros(3),)*6 + (to,))
        p._eval_field(fields, 1.0)
        self.assertTrue(np.allclose(p.Ex, [1.0, 2.0, 3.0]))  # t=1.0 + t_offset

    def test_t_offset_buffer_extend(self):
        """_extend preserves t_offset values for existing particles."""
        to = np.array([1.0, 2.0, 3.0])
        p = Particles('e', q=-1, m=1, N=3, props=(np.zeros(3),)*6 + (to,))
        p._extend(10)
        self.assertTrue(np.allclose(p.t_offset[:3], to))

    def test_t_offset_buffer_prune(self):
        """_prune preserves t_offset for remaining particles."""
        to = np.array([1.0, 2.0, 3.0])
        p = Particles('e', q=-1, m=1, N=3, props=(np.zeros(3),)*6 + (to,))
        p._to_be_pruned[1] = True
        p._prune()
        self.assertTrue(np.allclose(p.t_offset, [1.0, 3.0]))

    def test_t_offset_inheritance_photon(self):
        """Photon creation inherits t_offset from parent electron."""
        ele = Particles(
            'ele', q=-1, m=1, N=1,
            props=(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1)),
            RR=RadiationReactionType.PHOTON
        )
        to = np.array([5.0])
        ele.t_offset = to.copy()
        pho = Particles('pho', q=0, m=0)
        ele.set_photon(pho)
        # Manually trigger an event
        ele.event[0] = True
        ele.photon_delta[0] = 0.5
        ele._create_photon(pho)
        self.assertEqual(pho.N_buffered, 1)
        self.assertTrue(np.allclose(pho.t_offset[:1], to))

    def test_t_offset_inheritance_pair(self):
        """Pair creation inherits t_offset from parent photon."""
        pho = Particles(
            'pho', q=0, m=0, N=1,
            props=(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1)),
        )
        to = np.array([7.0])
        pho.t_offset = to.copy()
        ele = Particles('ele', q=-1, m=1)
        pos = Particles('pos', q=1, m=1)
        pho.set_pair(ele, pos)
        # Manually trigger an event
        pho.event[0] = True
        pho.pair_delta[0] = 0.5
        pho._create_pair(ele, pos)
        self.assertEqual(ele.N_buffered, 1)
        self.assertEqual(pos.N_buffered, 1)
        self.assertTrue(np.allclose(ele.t_offset[:1], to))
        self.assertTrue(np.allclose(pos.t_offset[:1], to))


if __name__ == '__main__':
    unittest.main()
