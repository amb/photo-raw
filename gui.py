from __future__ import absolute_import

import ctypes

import imgui
import numpy as np
import pyglet
from imgui.integrations.pyglet import create_renderer
from pyglet import gl


def generate_image_data(width, height, params):
    r_image = (np.zeros((width, height, 4)) * 255.0).astype("uint8")
    r_image[..., 3] = 255
    return r_image


def get_pyglet_rgba_texture(width, height, params):
    r_image = generate_image_data(width, height, params)
    r_image.reshape((width * height * 4,))
    c_glu_p = ctypes.POINTER(gl.GLubyte)
    gr_image = r_image.ctypes.data_as(c_glu_p)
    tx = pyglet.image.ImageData(width, height, "RGBA", gr_image, pitch=width * 4 * 1).get_texture()
    return tx


# ----- Globals

background_texture = None
any_changed = False
x_loc_sliders = None
y_loc_sliders = None
x_optics = None
y_optics = None
r_loc_sliders = None
p_fov = 140.0


def main(x_pos_sliders, y_pos_sliders, r_pos_sliders, pm_fov):
    window = pyglet.window.Window(width=1024, height=512, resizable=False)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    imgui.create_context()
    impl = create_renderer(window)

    global x_loc_sliders
    global y_loc_sliders
    global x_optics
    global y_optics
    global r_loc_sliders
    global p_fov
    x_loc_sliders = x_pos_sliders
    y_loc_sliders = y_pos_sliders
    r_loc_sliders = r_pos_sliders
    x_optics = [0] * len(x_loc_sliders)
    y_optics = [0] * len(x_loc_sliders)
    p_fov = pm_fov

    def update(dt):
        global background_texture
        global x_loc_sliders
        global y_loc_sliders
        global x_optics
        global y_optics
        global r_loc_sliders
        global any_changed
        global p_fov

        imgui.new_frame()
        io = imgui.get_io()

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
        imgui.image(background_texture.id, 1024.0, 512.0)
        imgui.end()
        imgui.pop_style_var()

        # ----- Foreground data windows

        any_changed = False

        imgui.begin(
            "Params window",
            False,
            flags=imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_HORIZONTAL_SCROLLING_BAR
        )
        # imgui.text("Bar")
        # imgui.same_line()
        # imgui.text_colored("Eggs", 0.2, 1.0, 0.0)

        # if imgui.button("Test"):
        #     print("Mouse:", io.mouse_pos)

        for sld_i, sld in enumerate(x_loc_sliders):
            change, x_loc_sliders[sld_i] = imgui.slider_float(
                f"X{sld_i}", x_loc_sliders[sld_i], 0.0, 1.0
            )
            any_changed |= change
            change, y_loc_sliders[sld_i] = imgui.slider_float(
                f"Y{sld_i}", y_loc_sliders[sld_i], -0.5, 0.5
            )
            any_changed |= change
            # change, x_optics[sld_i] = imgui.slider_float(f"Xo{sld_i}", x_optics[sld_i], -0.5, 0.5)
            # any_changed |= change
            # change, y_optics[sld_i] = imgui.slider_float(f"Yo{sld_i}", y_optics[sld_i], -0.5, 0.5)
            # any_changed |= change
            change, r_loc_sliders[sld_i] = imgui.slider_float(
                f"R{sld_i}", r_loc_sliders[sld_i], -0.5, 0.5
            )
            any_changed |= change
            imgui.dummy(0.0, 5.0)

        change, p_fov = imgui.slider_float("FOV", p_fov, 0.0, 360.0)
        any_changed |= change

        imgui.end()

    def draw(dt):
        global any_changed
        global background_texture
        global x_loc_sliders
        global y_loc_sliders
        global x_optics
        global y_optics
        global r_loc_sliders
        global p_fov

        if any_changed or background_texture is None:
            print(x_loc_sliders, y_loc_sliders, r_loc_sliders, p_fov)
            background_texture = get_pyglet_rgba_texture(
                1024,
                512,
                {
                    "x_locs": x_loc_sliders,
                    "y_locs": y_loc_sliders,
                    "x_optic": x_optics,
                    "y_optic": y_optics,
                    "rot": r_loc_sliders,
                    "fov": p_fov,
                },
            )

        update(dt)
        window.clear()
        imgui.render()
        impl.render(imgui.get_draw_data())

    pyglet.clock.schedule_interval(draw, 1 / 60.0)
    pyglet.app.run()
    impl.shutdown()

