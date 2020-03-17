import numpy as np
from refnx.reflect import Component, SLD, ReflectModel, Structure
from refnx.analysis import possibly_create_parameter, Parameters, Parameter


class monolayer(Component):

    def __init__(self, apm, b_heads, vm_heads, thickness_heads,
                 b_tails, vm_tails, thickness_tails, roughness, solvfrac = 0,
                 head_solvent=None, reverse_monolayer=False, name = ''):

        super(monolayer, self).__init__()
        self.apm = possibly_create_parameter(apm, 'area_per_molecule')

        if isinstance(b_heads, complex):
            self.b_heads_real = possibly_create_parameter(
                b_heads.real,
                name='b_heads_real')
            self.b_heads_imag = possibly_create_parameter(
                b_heads.imag,
                name='b_heads_imag')
        elif isinstance(b_heads, SLD):
            self.b_heads_real = b_heads.real
            self.b_heads_imag = b_heads.imag
        else:
            self.b_heads_real = possibly_create_parameter(
                b_heads,
                name='b_heads_real')
            self.b_heads_imag = possibly_create_parameter(
                0,
                name='b_heads_imag')

        self.vm_heads = possibly_create_parameter(
            vm_heads,
            name='vm_heads')

        self.thickness_heads = possibly_create_parameter(
            thickness_heads,
            name='thickness_heads')

        if isinstance(b_tails, complex):
            self.b_tails_real = possibly_create_parameter(
                b_tails.real,
                name='b_tails_real')
            self.b_tails_imag = possibly_create_parameter(
                b_tails.imag,
                name='b_tails_imag')
        elif isinstance(b_tails, SLD):
            self.b_tails_real = b_tails.real
            self.b_tails_imag = b_tails.imag
        else:
            self.b_tails_real = possibly_create_parameter(
                b_tails,
                name='b_tails_real')
            self.b_tails_imag = possibly_create_parameter(
                0,
                name='b_tails_imag')

        self.vm_tails = possibly_create_parameter(
            vm_tails,
            name='vm_tails')
        self.thickness_tails = possibly_create_parameter(
            thickness_tails,
            name='thickness_tails')

        self.roughness = possibly_create_parameter(
            roughness,
            name='roughness')

        self.head_solvent = None
        if head_solvent == 'd2o':
            self.head_solvent = 6.02316
        else:
            self.head_solvent = 2.3117

        self.solvfrac = possibly_create_parameter(solvfrac, 'solvent fraction')

        self.reverse_monolayer = reverse_monolayer
        self.name = name

    def __repr__(self):
        d = {}
        d.update(self.__dict__)
        sld_bh = SLD([self.b_heads_real, self.b_heads_imag])
        sld_bt = SLD([self.b_tails_real, self.b_tails_imag])
        d['bh'] = sld_bh
        d['bt'] = sld_bt

        s = (" Monolayer({apm!r}, {bh!r}, {vm_heads!r}, {thickness_heads!r},"
             " {bt!r}, {vm_tails!r}, {thickness_tails!r}, {roughness!r},"
             " head_solvent={head_solvent!r}, solvfrac={solvfrac!r}"
             " reverse_monolayer={reverse_monolayer}, name={name!r})")
        return s.format(**d)

    def slabs(self, structure=None):

        layers = np.zeros((2, 5))

        # thicknesses

        layers[0, 0] = float(self.thickness_heads)
        layers[1, 0] = float(self.thickness_tails)

        #Volume fraction of solvent
        volfrac = self.solvfrac.value

        # SLD's
        sld_heads = (1 - volfrac)*(float(self.b_heads_real) / float(self.vm_heads) * 1.e6) + volfrac*self.head_solvent

        layers[0, 1] = sld_heads
        layers[0, 2] = float(self.b_heads_imag) / float(self.vm_heads) * 1.e6

        layers[1, 1] = float(self.b_tails_real) / float(self.vm_tails) * 1.e6
        layers[1, 2] = float(self.b_tails_imag) / float(self.vm_tails) * 1.e6

        # roughnesses
        layers[0, 3] = float(self.roughness)
        layers[1, 3] = float(self.roughness)

        # volume fractions
        # head region

        layers[0, 4] = 0
        layers[1, 4] = 0

        if self.reverse_monolayer:
            layers = np.flipud(layers)
            layers[:, 3] = layers[::-1, 3]

        return layers


    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.apm, self.solvfrac,
                  self.b_heads_real, self.b_heads_imag, self.vm_heads,
                  self.thickness_heads,
                  self.b_tails_real, self.b_tails_imag, self.vm_tails,
                  self.thickness_tails, self.roughness])

        return p

    def logp(self):
        # penalise unphysical volume fractions.
        volfrac_h = self.vm_heads.value / (self.apm.value *
                                           self.thickness_heads.value)

        # tail region
        volfrac_t = self.vm_tails.value / (self.apm.value *
                                           self.thickness_tails.value)

        if volfrac_h > 1 or volfrac_t > 1:
            return -np.inf

        return 0
