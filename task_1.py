import numpy as np
from math import cos, sin
from manim import *
from helpers import get_mut_dot, c2p, get_line, get_vector, get_vec_info, get_dot_title, \
    get_vector_title, get_axes_config, to_norm_fun, get_mut_circle


class Task1(Scene):
    # Properties
    OO1 = AO2 = 20
    R = 18
    O1O2 = OA = 2 * R

    # Axes configs
    X_STEP = 5
    Y_STEP = 5
    X_RANGE = [-1.1 * OO1, 1.2 * (OO1 + O1O2), X_STEP]
    Y_RANGE = [-1.1 * (OO1 + R), 1.1 * (OO1 + R), Y_STEP]
    X_LENGTH = 7
    AXES_NUMS_FONT_SIZE = 16
    AXES_LABELS_FONT_SIZE = 28
    AXES_CONFIG = get_axes_config(X_RANGE, Y_RANGE, X_STEP, Y_STEP, AXES_NUMS_FONT_SIZE, X_LENGTH)

    # Vector configs
    VECTORS_KWARGS = {
        'stroke_width': 3,
        'tip_length': 0.25,
        'buff': 0,
        'max_tip_length_to_length_ratio': 0.2,
        'max_stroke_width_to_length_ratio': 10,
    }

    FIXED_POINTS = {
        'O1': np.array((0, 0, 0)),
        'O2': np.array((O1O2, 0, 0)),
    }

    # Simulation time
    INITIAL_TIME = 0
    END_TIME = pow(12, 1 / 3)

    # Playback speed
    P_SPEED = 1 / 2.5

    TITLES_FONT = 18
    TITLES_FONT_2 = 22
    LINE_WIDTH = DEFAULT_STROKE_WIDTH / 1.5

    VEL_SCALE = 0.2  # Velocity scale
    ACC_SCALE = 0.07  # Acceleration scale

    @staticmethod
    def phi(t: float) -> float:
        return PI * t ** 3 / 6

    @staticmethod
    def omega(t: float) -> float:
        return PI * t ** 2 / 2

    @staticmethod
    def epsilon(t: float) -> float:
        return PI * t

    @staticmethod
    def om(t: float) -> float:
        return 6 * PI * t ** 2

    @staticmethod
    def om_v(t: float) -> float:
        return 12 * PI * t

    @staticmethod
    def om_a(t: float) -> float:
        return 12 * PI

    @staticmethod
    def o_coord(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.OO1 * cos(source.phi(t)),
                source.OO1 * sin(source.phi(t)),
                0,
            )
        )

    @staticmethod
    def o_vel(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                -source.OO1 * sin(source.phi(t)) * source.omega(t),
                source.OO1 * cos(source.phi(t)) * source.omega(t),
                0,
            )
        )

    @staticmethod
    def o_acc(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                -source.OO1 * (cos(source.phi(t)) * (source.omega(t) ** 2) + sin(source.phi(t)) * source.epsilon(t)),
                source.OO1 * (cos(source.phi(t)) * source.epsilon(t) - sin(source.phi(t)) * (source.omega(t) ** 2)),
                0,
            )
        )

    @staticmethod
    def a_coord(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.O1O2 + source.AO2 * cos(source.phi(t)),
                source.AO2 * sin(source.phi(t)),
                0,
            )
        )

    @staticmethod
    def m_coord_r(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.R - source.R * cos(source.om(t) / source.R),
                source.R * sin(source.om(t) / source.R),
                0,
            )
        )

    @staticmethod
    def m_coord(t: float) -> np.ndarray:
        source = Task1
        return source.o_coord(t) + source.m_coord_r(t)

    @staticmethod
    def m_vel_r(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.om_v(t) * sin(source.om(t) / source.R),
                source.om_v(t) * cos(source.om(t) / source.R),
                0,
            )
        )

    @staticmethod
    def m_vel_t(t: float) -> np.ndarray:
        source = Task1
        return source.o_vel(t)

    @staticmethod
    def m_vel_a(t: float) -> np.ndarray:
        source = Task1
        return source.m_vel_t(t) + source.m_vel_r(t)

    @staticmethod
    def m_acc_r(t: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.om_a(t) * sin(source.om(t) / source.R) +
                ((source.om_v(t) ** 2) * cos(source.om(t) / source.R)) / source.R,
                source.om_a(t) * cos(source.om(t) / source.R) -
                ((source.om_v(t) ** 2) * sin(source.om(t) / source.R)) / source.R,
                0,
            )
        )

    @staticmethod
    def m_acc_c(t: float) -> np.ndarray:
        return np.array((0, 0, 0))

    @staticmethod
    def m_acc_t(t: float) -> np.ndarray:
        source = Task1
        return source.o_acc(t)

    @staticmethod
    def m_acc_a(t: float):
        source = Task1
        return source.m_acc_t(t) + source.m_acc_c(t) + source.m_acc_r(t)

    def get_dots(self, t: ValueTracker, axes: Axes):
        laws = [self.o_coord, self.a_coord, self.m_coord]
        radius = DEFAULT_DOT_RADIUS / 3
        c = WHITE

        return [
                   Dot(c2p(axes, p), radius=radius, color=c)
                   for p in self.FIXED_POINTS.values()
               ] + [
                   get_mut_dot(t, axes, c, radius, law)
                   for law in laws
               ]

    def get_dot_titles(self, t: ValueTracker, axes: Axes):
        titles = ['O_1', 'O_2', 'O', 'A', 'M']
        laws = [self.o_coord, self.a_coord, self.m_coord]

        shift_laws = [lambda ct: np.array((-0.15, -0.15, 0))] * len(titles)
        shift_laws[1] = shift_laws[-1] = lambda ct: np.array((0.15, -0.15, 0))

        return [
                   MathTex(title, font_size=self.TITLES_FONT).move_to(c2p(axes, p) + shift(0))
                   for title, p, shift in zip(titles[:2], self.FIXED_POINTS.values(), shift_laws[:2])
               ] + [
                   get_dot_title(t, axes, title, self.TITLES_FONT, law, shift_law)
                   for title, law, shift_law in zip(titles[2:], laws, shift_laws[2:])
               ]

    def get_lines(self, t: ValueTracker, axes: Axes):
        laws = [
            (lambda ct: self.FIXED_POINTS['O1'], self.o_coord),
            (lambda ct: self.FIXED_POINTS['O2'], self.a_coord),
            (self.o_coord, self.a_coord),
        ]
        c = GREEN
        return (
            get_line(t, axes, c, s_law, e_law, stroke_width=self.LINE_WIDTH)
            for s_law, e_law in laws
        )

    def get_vectors(self, t: ValueTracker, axes: Axes, e_functions: list, colors: list, scale: float):
        s_functions = [self.m_coord, self.m_coord, self.m_coord]
        return (
            get_vector(t, axes, c, s_fun, e_fun, scale, self.VECTORS_KWARGS)
            for c, s_fun, e_fun in zip(colors, s_functions, e_functions)
        )

    def get_all_vel(self, t: ValueTracker, axes: Axes):
        e_functions = [self.m_vel_a, self.m_vel_t, self.m_vel_r]
        colors = [BLUE_E, BLUE_D, BLUE_A]
        return self.get_vectors(t, axes, e_functions, colors, self.VEL_SCALE)

    def get_all_acc(self, t: ValueTracker, axes: Axes):
        e_functions = [self.m_acc_a, self.m_acc_t, self.m_acc_r]
        colors = [RED_E, RED_B, RED_A]
        return self.get_vectors(t, axes, e_functions, colors, self.ACC_SCALE)

    def get_titles(self, t: ValueTracker, titles: list[str], vectors: list[Arrow]):
        shift_laws = [
                         lambda vec: normalize(vec) * 0.13
                     ] * len(vectors)

        return (
            get_vector_title(t, vector, title, self.TITLES_FONT, shift_law)
            for vector, title, shift_law in zip(vectors, titles, shift_laws)
        )

    @staticmethod
    def get_dot_trajectory(axes: Axes, law: Callable[[float], np.ndarray], t_range: np.ndarray, num_dashes: int):
        curve = axes.plot_parametric_curve(
            law, color=WHITE, t_range=t_range,
            stroke_width=DEFAULT_STROKE_WIDTH / 5
        )
        return DashedVMobject(curve, num_dashes=num_dashes)

    def plot_graphs(self, axes: Axes, laws: list[Callable[[float], np.ndarray]], colors: list[str]):
        return [
            axes.plot(
                to_norm_fun(law),
                np.array([self.INITIAL_TIME, self.END_TIME]),
                stroke_width=DEFAULT_STROKE_WIDTH / 4,
                color=c,
            )
            for law, c in zip(laws, colors)
        ]

    def create_graphs(self):
        graphs_font_size = 15
        size = 3.2

        axes_1_pos = 4.5 * RIGHT + 2 * UP
        axes_1 = Axes(
            **get_axes_config(
                [0, self.END_TIME * 1.12, 0.2], [0, 270, 25],
                None, None, graphs_font_size, size, size, x_decimal_place=1
            )
        ).move_to(axes_1_pos)
        axes_1_labels = axes_1.get_axis_labels(
            MathTex('t,s', font_size=graphs_font_size),
            MathTex('V,\\frac{m}{s}', font_size=graphs_font_size),
        )

        axes_2_pos = 4.5 * RIGHT + 2 * DOWN
        axes_2 = Axes(
            **get_axes_config(
                [0, self.END_TIME * 1.12, 0.2], [0, 1500, 125],
                None, None, graphs_font_size, size, size, x_decimal_place=1
            )
        ).move_to(axes_2_pos)
        axes_2_labels = axes_2.get_axis_labels(
            MathTex('t,s', font_size=graphs_font_size),
            MathTex('a,\\frac{m}{s^2}', font_size=graphs_font_size),
        )

        abs_vg, tr_vg, rel_vg = self.plot_graphs(
            axes_1, [self.m_vel_a, self.m_vel_t, self.m_vel_r], [BLUE_E, BLUE_D, BLUE_A]
        )
        abs_ag, tr_ag, rel_ag = self.plot_graphs(
            axes_2, [self.m_acc_a, self.m_acc_t, self.m_acc_r], [RED_E, RED_B, RED_A]
        )

        return axes_1, axes_1_labels, axes_2, axes_2_labels, abs_vg, tr_vg, rel_vg, abs_ag, tr_ag, rel_ag

    @staticmethod
    def create_graph_law(law: Callable[[float], np.ndarray]) -> Callable[[float], np.ndarray]:
        return lambda ct: np.array(
            (ct, to_norm_fun(law)(ct), 0)
        )

    def create_graph_dots(self, t: ValueTracker, axes: Axes,
                          laws: list[Callable[[float], np.ndarray]], colors: list[str]):
        radius = DEFAULT_DOT_RADIUS / 3
        return [
            get_mut_dot(t, axes, c, radius, self.create_graph_law(law))
            for c, law in zip(colors, laws)
        ]

    def create_graph_dots_labels(self, t: ValueTracker, axes: Axes,
                                 titles: list[str], font_size: int,
                                 laws: list[Callable[[float], np.ndarray]],
                                 colors: list[str]):
        shift_laws = [
            lambda ct: np.array((0, 0.17, 0)),
            lambda ct: np.array((0.17, 0, 0)),
            lambda ct: np.array((0, -0.17, 0))
        ]
        return [
            get_dot_title(t, axes, title, font_size, self.create_graph_law(law), shift_law, c)
            for title, law, shift_law, c in zip(titles, laws, shift_laws, colors)
        ]

    def create_graph_1_dots(self, t: ValueTracker, axes: Axes):
        laws = [self.m_vel_a, self.m_vel_t, self.m_vel_r]
        colors = [BLUE_E, BLUE_D, BLUE_A]
        return self.create_graph_dots(t, axes, laws, colors)

    def create_graph_2_dots(self, t: ValueTracker, axes: Axes):
        laws = [self.m_acc_a, self.m_acc_t, self.m_acc_r]
        colors = [RED_E, RED_B, RED_A]
        return self.create_graph_dots(t, axes, laws, colors)

    def create_graph_1_dots_labels(self, t: ValueTracker, axes: Axes):
        laws = [self.m_vel_a, self.m_vel_t, self.m_vel_r]
        titles = ['V_a', 'V_{tr}', 'V_r']
        colors = [BLUE_E, BLUE_D, BLUE_A]
        return self.create_graph_dots_labels(t, axes, titles, self.TITLES_FONT_2, laws, colors)

    def create_graph_2_dots_labels(self, t: ValueTracker, axes: Axes):
        laws = [self.m_acc_a, self.m_acc_t, self.m_acc_r]
        titles = ['a_a', 'a_{tr}', 'a_r']
        colors = [RED_E, RED_B, RED_A]
        return self.create_graph_dots_labels(t, axes, titles, self.TITLES_FONT_2, laws, colors)

    def construct(self):
        t = ValueTracker(self.INITIAL_TIME)

        axes_pos = LEFT * 3.25
        axes = Axes(**self.AXES_CONFIG).move_to(axes_pos)
        axes_labels = axes.get_axis_labels(
            MathTex('x', font_size=self.AXES_LABELS_FONT_SIZE),
            MathTex('y', font_size=self.AXES_LABELS_FONT_SIZE),
        )

        o1_dot, o2_dot, o_dot, a_dot, m_dot = self.get_dots(t, axes)
        o1_t_dot, o2_t_dot, o_t_dot, a_t_dot, m_t_dot = self.get_dot_titles(t, axes)
        oo1_l, ao2_l, oa_l = self.get_lines(t, axes)

        abs_v, tr_v, rel_v = self.get_all_vel(t, axes)
        abs_a, tr_a, rel_a = self.get_all_acc(t, axes)
        abs_vt, tr_vt, rel_vt, abs_at, tr_at, rel_at = self.get_titles(
            t,
            ['\\Vec{V}_a', '\\Vec{V}_{tr}', '\\Vec{V}_r', '\\Vec{a}_a', '\\Vec{a}_{tr}', '\\Vec{a}_r'],
            [abs_v, tr_v, rel_v, abs_a, tr_a, rel_a]
        )

        circle = get_mut_circle(
            t, axes, self.R, GREEN, self.LINE_WIDTH,
            lambda ct: (self.o_coord(ct) + self.a_coord(ct)) / 2,
            fill_opacity=0.4
        )

        m_path = self.get_dot_trajectory(axes, self.m_coord, np.array([self.INITIAL_TIME, self.END_TIME]), 30)

        axes_1, axes_1_labels, axes_2, axes_2_labels, abs_vg, tr_vg, rel_vg, abs_ag, tr_ag, rel_ag = \
            self.create_graphs()

        abs_vg_d, tr_vg_d, rel_vg_d = self.create_graph_1_dots(t, axes_1)
        abs_ag_d, tr_ag_d, rel_ag_d = self.create_graph_2_dots(t, axes_2)

        abs_vg_l, tr_vg_l, rel_vg_l = self.create_graph_1_dots_labels(t, axes_1)
        abs_ag_l, tr_ag_l, rel_ag_l = self.create_graph_2_dots_labels(t, axes_2)

        timer_pos = UP * 3.6
        timer = get_vec_info(
            t, 't=', self.TITLES_FONT, lambda c_t: c_t, {'num_decimal_places': 3, 'font_size': self.TITLES_FONT}
        ).arrange(RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER / 5).move_to(timer_pos)

        self.add(
            axes, axes_labels, m_path,
            axes_1, axes_1_labels, axes_2, axes_2_labels,
            abs_vg, tr_vg, rel_vg, abs_ag, tr_ag, rel_ag,
            abs_vg_d, tr_vg_d, rel_vg_d, abs_ag_d, tr_ag_d, rel_ag_d,
            abs_vg_l, tr_vg_l, rel_vg_l, abs_ag_l, tr_ag_l, rel_ag_l,
            o1_t_dot, o2_t_dot, o_t_dot, a_t_dot, m_t_dot,
            oo1_l, ao2_l, oa_l,
            circle,
            abs_v, tr_v, rel_v,
            abs_a, tr_a, rel_a,
            abs_vt, tr_vt, rel_vt, abs_at, tr_at, rel_at,
            timer,
            o1_dot, o2_dot, o_dot, a_dot, m_dot,
        )
        self.play(
            t.animate.set_value(self.END_TIME),
            run_time=(self.END_TIME - self.INITIAL_TIME) / self.P_SPEED,
            rate_func=linear
        )
