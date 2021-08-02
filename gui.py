from __future__ import absolute_import

import ctypes

import imgui
import numpy as np
import pyglet
from imgui.integrations.pyglet import create_renderer, PygletRenderer
from pyglet import gl


def generate_image_data(width, height, params):
    r_image = (np.zeros((width, height, 4)) * 255.0).astype("uint8")
    r_image[..., 3] = 255
    return r_image


def numpy_to_gl(arr):
    assert len(arr.shape) == 3
    assert arr.dtype == np.uint8
    arr = arr.reshape((arr.shape[0] * arr.shape[1] * 4,))
    c_glu_p = ctypes.POINTER(gl.GLubyte)
    return arr.ctypes.data_as(c_glu_p)


def get_pyglet_rgba_texture(width, height, params, tex):
    out_cl = generate_image_data(width, height, params)

    final = out_cl.to_numpy()[::2, ::2, :]
    final[..., 3] = 1.0

    # out_d = (gl.GLubyte * r_image.size).from_buffer(r_image)
    tex.set_data("RGBA", width * 4, numpy_to_gl((final * 255.0).astype("uint8")))


class UI:
    def __init__(self, window):
        imgui.create_context()
        self.impl = PygletRenderer(window)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.p_fov = 120.0
        self.x_loc_sliders = [0.0]
        self.y_loc_sliders = [0.0]
        self.r_loc_sliders = [0.0]
        self.x_optics = [0] * len(self.x_loc_sliders)
        self.y_optics = [0] * len(self.x_loc_sliders)

        self.width = window.width
        self.height = window.height

        # Set background texture to black
        self.background_texture = pyglet.image.ImageData(
            self.width,
            self.height,
            "RGBA",
            numpy_to_gl(np.zeros((self.width, self.height, 4)).astype(np.uint8)),
            pitch=self.width * 4,
        )

        self.bg_id = self.background_texture.get_texture().id
        print("bg:", self.background_texture.pitch, self.background_texture.format)

        self.any_changed = False

    def render(self):

        io = imgui.get_io()

        imgui.new_frame()

        # ----- Background image plane

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*io.display_size)
        imgui.set_next_window_content_size(1024.0, 512.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, imgui.Vec2(0, 0))
        imgui.begin(
            "Full frame window",
            False,
            flags=imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_INPUTS
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_SCROLL_WITH_MOUSE
            | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
            | imgui.WINDOW_NO_TITLE_BAR,
        )
        imgui.image(self.background_texture.get_texture().id, 1024.0, 512.0)
        imgui.end()
        imgui.pop_style_var()

        # ----- Foreground data windows

        any_changed = False

        imgui.begin(
            "Params window",
            False,
            flags=imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_HORIZONTAL_SCROLLING_BAR,
        )

        for sld_i, sld in enumerate(self.x_loc_sliders):
            change, self.x_loc_sliders[sld_i] = imgui.slider_float(
                f"X{sld_i}", self.x_loc_sliders[sld_i], 0.0, 1.0
            )
            any_changed |= change
            change, self.y_loc_sliders[sld_i] = imgui.slider_float(
                f"Y{sld_i}", self.y_loc_sliders[sld_i], -0.5, 0.5
            )
            any_changed |= change
            # change, x_optics[sld_i] = imgui.slider_float(f"Xo{sld_i}", x_optics[sld_i], -0.5, 0.5)
            # any_changed |= change
            # change, y_optics[sld_i] = imgui.slider_float(f"Yo{sld_i}", y_optics[sld_i], -0.5, 0.5)
            # any_changed |= change
            change, self.r_loc_sliders[sld_i] = imgui.slider_float(
                f"R{sld_i}", self.r_loc_sliders[sld_i], -0.5, 0.5
            )
            any_changed |= change
            imgui.dummy(0.0, 5.0)

        change, self.p_fov = imgui.slider_float("FOV", self.p_fov, 0.0, 360.0)
        any_changed |= change
        self.any_changed = any_changed

        imgui.end()

        imgui.end_frame()

        imgui.render()
        self.impl.render(imgui.get_draw_data())

        if self.any_changed:
            # print("redraw")
            get_pyglet_rgba_texture(
                1024,
                512,
                {
                    "x_locs": self.x_loc_sliders,
                    "y_locs": self.y_loc_sliders,
                    "x_optic": self.x_optics,
                    "y_optic": self.y_optics,
                    "rot": self.r_loc_sliders,
                    "fov": self.p_fov,
                },
                self.background_texture,
            )
            # self.background_texture.set_data(
            #     "RGBA",
            #     self.width * 4,
            #     numpy_to_gl(
            #         np.random.random_integers(255, size=(self.width, self.height, 4)).astype(
            #             np.uint8
            #         )
            #     ),
            # )


class App(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=1024, height=512, resizable=False)
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        self.UI_test = UI(self)

    def on_draw(self):
        pass

    def update(self, dt):
        # self.clear()
        self.UI_test.render()


def run():
    app = App()
    pyglet.app.run()
