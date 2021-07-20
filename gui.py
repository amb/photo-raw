# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pyglet
from pyglet import gl

import imgui

# Note that we could explicitly choose to use PygletFixedPipelineRenderer
# or PygletProgrammablePipelineRenderer, but create_renderer handles the
# version checking for us.
from imgui.integrations.pyglet import create_renderer


import numpy as np
import ctypes


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
position_sliders = None


def main(pos_sliders):
    window = pyglet.window.Window(width=1024, height=512, resizable=False)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    imgui.create_context()
    impl = create_renderer(window)

    global position_sliders
    position_sliders = pos_sliders

    def update(dt):
        global background_texture
        global position_sliders
        global any_changed

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
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_TITLE_BAR,
        )
        # imgui.text("Bar")
        # imgui.same_line()
        # imgui.text_colored("Eggs", 0.2, 1.0, 0.0)

        # if imgui.button("Test"):
        #     print("Mouse:", io.mouse_pos)

        for sld_i, sld in enumerate(position_sliders):
            # if imgui.button("<"):
            #     position_sliders[sld_i] -= 0.002
            #     print(sld_i)
            # imgui.same_line()
            # if imgui.button(">"):
            #     position_sliders[sld_i] += 0.002
            #     print(sld_i)
            # imgui.same_line()
            # if position_sliders[sld_i] > 1.0:
            #     position_sliders[sld_i] = 1.0
            # if position_sliders[sld_i] < 0.0:
            #     position_sliders[sld_i] = 0.0

            change, position_sliders[sld_i] = imgui.slider_float(
                f"X{sld_i}", position_sliders[sld_i], 0.0, 1.0
            )
            if change:
                print(position_sliders[sld_i])

            any_changed |= change

        imgui.end()

    def draw(dt):
        global any_changed
        global background_texture
        global position_sliders

        if any_changed or background_texture is None:
            background_texture = get_pyglet_rgba_texture(1024, 512, {"x_locs": position_sliders})

        update(dt)
        window.clear()

        imgui.render()

        impl.render(imgui.get_draw_data())

    pyglet.clock.schedule_interval(draw, 1 / 60.0)
    pyglet.app.run()
    impl.shutdown()


# if __name__ == "__main__":
#     main()
