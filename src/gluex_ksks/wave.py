from __future__ import annotations
import re
from itertools import combinations
from typing import override

import laddu as ld

from gluex_ksks.constants import BLUE, RED


class Wave:
    def __init__(self, l: int, m: int, r: str) -> None:  # noqa: E741
        assert l in {0, 1, 2, 3}, 'L < 0 or L > 3 is not supported!'
        assert -l <= m <= l
        assert r in {'-', '+'}
        self.l: int = l  # noqa: E741
        self.m: int = m
        self.r: str = r

    @override
    def __repr__(self) -> str:
        return f'Wave({self.l}, {self.m}, {self.r})'

    def __lt__(self, other: Wave) -> bool:
        return (self.l, self.m, self.r) < (other.l, other.m, other.r)

    def encode(self) -> int:
        return (self.l << 4) | ((self.m + 3) << 1) | (1 if self.positive else 0)

    @staticmethod
    def decode(code: int) -> Wave:
        return Wave((code >> 4) & 3, ((code >> 1) & 7) - 3, '+' if code & 1 else '-')

    @staticmethod
    def encode_waves(waves: list[Wave]) -> int:
        waves = list(sorted(waves))
        encoded = 0
        for wave in waves:
            encoded = (encoded << 6) | wave.encode()
        return encoded

    @staticmethod
    def decode_waves(encoded: int) -> list[Wave]:
        waves = []
        while encoded:
            wave_data = encoded & 0b111111
            waves.append(Wave.decode(wave_data))
            encoded >>= 6
        return list(sorted(waves))

    @staticmethod
    def to_latex_string(waves: list[Wave]) -> str:
        waves = list(sorted(waves))
        if len(waves) == 1:
            return waves[0].latex
        if len(waves) == 2:
            return f'{waves[0].latex} and {waves[1].latex}'
        return (
            f'{", ".join([wave.latex for wave in waves[:-1]])}, and {waves[-1].latex}'
        )

    @staticmethod
    def to_kebab_string(waves: list[Wave]) -> str:
        waves = list(sorted(waves))
        return '-'.join(
            f'{wave.letter}{"p" if wave.m >= 0 else "m"}{abs(wave.m)}{"p" if wave.positive else "m"}'
            for wave in waves
        )

    @property
    def plot_color(self) -> str:
        if self.positive:
            return RED
        else:
            return BLUE

    def plot_index(self, *, double: bool = False) -> tuple[int, int]:
        if double:
            if self.l == 0:
                return (0, 0)
            else:
                return (1, 0)
        if self.m == 0:
            if self.l == 0:
                return (0, 0)
            else:
                return (1, 0)
        if self.m > 0:
            return (0, self.m)
        else:
            return (1, abs(self.m))

    @staticmethod
    def needs_full_plot(waves: list[Wave]) -> bool:
        return len({wave.m for wave in waves if wave.l == 2}) > 1

    @staticmethod
    def has_wave_at_index(
        waves: list['Wave'], index: tuple[int, int], *, double: bool = False
    ) -> bool:
        for wave in waves:
            if wave.plot_index(double=double) == index:
                return True
        return False

    @staticmethod
    def get_latex_group_at_index(index: tuple[int, int]) -> str:
        m = index[1]
        if m == 0:
            sign = 1
            l = [0, 2][index[0]]  # noqa: E741
        else:
            sign = -(index[0] * 2 - 1)
            l = 2  # noqa: E741
        wave = Wave(l, m * sign, '+')
        return wave.latex_group

    @property
    def positive(self) -> bool:
        return self.r == '+'

    @property
    def negative(self) -> bool:
        return self.r == '-'

    @staticmethod
    def from_string(string: str) -> Wave:
        m = re.match(r'^(\d)([+-]?\d)([+-])$', string)
        if not m:
            raise Exception(f'The string {string} is not a valid wave!')
        return Wave(int(m.group(1)), int(m.group(2)), m.group(3))

    @property
    def zlm_name(self) -> str:
        return f'z{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def coefficient_name(self) -> str:
        return f'c{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def kmatrix_names(self) -> list[str]:
        return [self.kmatrix_name_f, self.kmatrix_name_a]

    @property
    def kmatrix_name_f(self) -> str:
        return f'f{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def kmatrix_name_a(self) -> str:
        return f'a{self.l}{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'

    @property
    def letter(self) -> str:
        match self.l:
            case 0:
                return 'S'
            case 1:
                return 'P'
            case 2:
                return 'D'
            case 3:
                return 'F'
            case _:
                raise Exception('L > 3 is not supported!')

    @staticmethod
    def power_set(waves: list[Wave]) -> list[list[Wave]]:
        return [
            list(sorted(subset))
            for r in range(1, len(waves) + 1)
            for subset in combinations(waves, r)
        ]

    @staticmethod
    def get_amplitude_names(
        wave: Wave, *, mass_dependent: bool, phase_factor: bool
    ) -> list[str]:
        names: list[str] = []
        names.append(wave.zlm_name)
        if mass_dependent:
            names.extend(wave.kmatrix_names)
        else:
            names.append(wave.coefficient_name)
        if phase_factor:
            names.append('kappa')
            names.append('k_scale')
        return names

    @staticmethod
    def get_waveset_names(
        waveset: list[Wave], *, mass_dependent: bool, phase_factor: bool
    ) -> list[str]:
        return list(
            list(
                set(
                    name
                    for wave in waveset
                    for name in Wave.get_amplitude_names(
                        wave,
                        mass_dependent=mass_dependent,
                        phase_factor=phase_factor,
                    )
                )
            )
        )

    @property
    def latex(self) -> str:
        return f'${self.letter}_{{{"+" if self.m > 0 else ("-" if self.m < 0 else "")}{abs(self.m)}}}^{{({self.r})}}$'

    @property
    def latex_group(self) -> str:
        return f'${self.letter}_{{{"+" if self.m > 0 else ("-" if self.m < 0 else "")}{abs(self.m)}}}$'

    def zlm(self, manager: ld.Manager) -> ld.amplitudes.AmplitudeID:
        angles = ld.Angles(0, [1], [2], [2, 3])
        polarization = ld.Polarization(0, [1], 0)
        return manager.register(
            ld.Zlm(
                self.zlm_name,
                self.l,
                self.m,
                self.r,
                angles,
                polarization,
            )
        )

    def coefficient(self, manager: ld.Manager) -> ld.amplitudes.AmplitudeID:
        if self.l == 0:
            return manager.register(
                ld.Scalar(
                    self.coefficient_name,
                    ld.parameter(f'{self.coefficient_name} real'),
                )
            )
        else:
            return manager.register(
                ld.ComplexScalar(
                    self.coefficient_name,
                    ld.parameter(f'{self.coefficient_name} real'),
                    ld.parameter(f'{self.coefficient_name} imag'),
                )
            )

    @staticmethod
    def kmatrix_parameter_name_to_latex_parts(par_name: str) -> tuple[str, str]:
        name, _ = par_name.split(' ')
        isospin = name[0]
        spin = int(name[1])
        mass = name[2:].split(')(')[0].replace('(', '')
        prime = "'" if isospin == 'f' and spin == 2 and mass == '1525' else ''
        subinfo = name[2:].split(')(')[1].replace(')', '')
        m = int(subinfo[:-1])
        r = subinfo[-1]
        letter = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}.get(spin, '?')
        wave = (
            f'{letter}_{{{"+" if m > 0 else ("-" if m < 0 else "")}{abs(m)}}}^{{({r})}}'
        )
        return (rf'${isospin}_{{{spin}}}{prime}({mass})$', rf'${wave}$')

    def kmatrix(
        self, manager: ld.Manager, seed: int | None = None
    ) -> list[ld.amplitudes.AmplitudeID]:
        res_mass = ld.Mass([2, 3])
        wave_subinfo = f'{"+" if self.m >= 0 else "-"}{abs(self.m)}{self.r}'
        if self.l == 0:
            f = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixF0(
                    self.kmatrix_name_f,
                    (
                        (ld.constant(0), ld.constant(0)),
                        (
                            ld.parameter(f'f0(980)({wave_subinfo}) real'),
                            ld.constant(0),
                        ),
                        (
                            ld.parameter(f'f0(1370)({wave_subinfo}) real'),
                            ld.parameter(f'f0(1370)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f0(1500)({wave_subinfo}) real'),
                            ld.parameter(f'f0(1500)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f0(1710)({wave_subinfo}) real'),
                            ld.parameter(f'f0(1710)({wave_subinfo}) imag'),
                        ),
                    ),
                    2,
                    res_mass,
                    seed=seed,
                )
            )
            a = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixA0(
                    self.kmatrix_name_a,
                    (
                        (
                            ld.parameter(f'a0(980)({wave_subinfo}) real'),
                            ld.parameter(f'a0(980)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'a0(1450)({wave_subinfo}) real'),
                            ld.parameter(f'a0(1450)({wave_subinfo}) imag'),
                        ),
                    ),
                    1,
                    res_mass,
                    seed=seed,
                )
            )
        elif self.l == 2:
            f = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixF2(
                    self.kmatrix_name_f,
                    (
                        (
                            ld.parameter(f'f2(1270)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1270)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f2(1525)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1525)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f2(1810)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1810)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'f2(1950)({wave_subinfo}) real'),
                            ld.parameter(f'f2(1950)({wave_subinfo}) imag'),
                        ),
                    ),
                    2,
                    res_mass,
                    seed=seed,
                )
            )
            a = manager.register(
                ld.amplitudes.kmatrix.KopfKMatrixA2(
                    self.kmatrix_name_a,
                    (
                        (
                            ld.parameter(f'a2(1320)({wave_subinfo}) real'),
                            ld.parameter(f'a2(1320)({wave_subinfo}) imag'),
                        ),
                        (
                            ld.parameter(f'a2(1700)({wave_subinfo}) real'),
                            ld.parameter(f'a2(1700)({wave_subinfo}) imag'),
                        ),
                    ),
                    1,
                    res_mass,
                    seed=seed,
                )
            )
        else:
            raise Exception('K-Matrix for L != 0, 2 is not supported!')
        return [f, a]

    @staticmethod
    def phase_space_factor(manager: ld.Manager) -> ld.amplitudes.AmplitudeID:
        m_resonance = ld.Mass([2, 3])
        m_1 = ld.Mass([2])
        m_2 = ld.Mass([3])
        m_recoil = ld.Mass([1])
        s = ld.Mandelstam([0], [], [2, 3], [1], channel='s')
        return manager.register(
            ld.PhaseSpaceFactor('kappa', m_recoil, m_1, m_2, m_resonance, s)
        )

    @staticmethod
    def get_model(
        waves: list[Wave],
        *,
        mass_dependent: bool,
        phase_factor: bool = False,
        resample_seed: int | None = None,
    ) -> ld.Model:
        pos_r_waves = [wave for wave in waves if wave.positive]
        neg_r_waves = [wave for wave in waves if wave.negative]
        manager = ld.Manager()
        pos_amps = (
            [wave.coefficient(manager) for wave in pos_r_waves]
            if not mass_dependent
            else [
                ld.amplitude_sum(wave.kmatrix(manager, seed=resample_seed))
                for wave in pos_r_waves
            ]
        )
        neg_amps = (
            [wave.coefficient(manager) for wave in neg_r_waves]
            if not mass_dependent
            else [ld.amplitude_sum(wave.kmatrix(manager)) for wave in neg_r_waves]
        )
        if phase_factor:
            k = Wave.phase_space_factor(manager) * manager.register(
                ld.Scalar('k_scale', ld.constant(1e3))
            )
            pos_amps = [k * a for a in pos_amps]
            neg_amps = [k * a for a in neg_amps]
        pos_zlms = [wave.zlm(manager) for wave in pos_r_waves]
        neg_zlms = [wave.zlm(manager) for wave in neg_r_waves]
        pos_re_terms = [c * zlm.real() for c, zlm in zip(pos_amps, pos_zlms)]
        pos_im_terms = [c * zlm.imag() for c, zlm in zip(pos_amps, pos_zlms)]
        neg_re_terms = [c * zlm.real() for c, zlm in zip(neg_amps, neg_zlms)]
        neg_im_terms = [c * zlm.imag() for c, zlm in zip(neg_amps, neg_zlms)]
        pos_re_sum = ld.amplitude_sum(pos_re_terms).norm_sqr()
        pos_im_sum = ld.amplitude_sum(pos_im_terms).norm_sqr()
        neg_re_sum = ld.amplitude_sum(neg_re_terms).norm_sqr()
        neg_im_sum = ld.amplitude_sum(neg_im_terms).norm_sqr()
        expr = pos_re_sum + pos_im_sum + neg_re_sum + neg_im_sum
        return manager.model(expr)
