# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pyglet
from pyglet import gl

import imgui

# Note that we could explicitly choose to use PygletFixedPipelineRenderer
# or PygletProgrammablePipelineRenderer, but create_renderer handles the
# version checking for us.
from imgui.integrations.pyglet import create_renderer


position_sliders = [0.1, 0.2, 0.3, 0.4]

import numpy as np


def random_pyglet_rgba_texture(width, height):
    r_image = (np.random.random((width, height, 4)) * 255.0).astype("uint8")
    r_image[..., 3] = 255
    r_image = r_image.flatten()
    gr_image = (gl.GLubyte * r_image.size)(*r_image)
    tx = pyglet.image.ImageData(width, height, "RGBA", gr_image, pitch=width * 4 * 1).get_texture()
    return tx


pyg_texture = random_pyglet_rgba_texture(1024, 512)


def main():
    window = pyglet.window.Window(width=1024, height=512, resizable=False)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    imgui.create_context()
    impl = create_renderer(window)

    def update(dt):
        imgui.new_frame()
        io = imgui.get_io()

        # ----- Background image plane

        imgui.set_next_window_position(-5, -5)
        imgui.set_next_window_size(*io.display_size)
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
        imgui.image(pyg_texture.id, 1024.0, 512.0)
        imgui.end()

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

        global position_sliders
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
                f"{sld_i}", position_sliders[sld_i], 0.0, 1.0
            )
            if change:
                print(position_sliders[sld_i])

            any_changed |= change

        if any_changed:
            print("change")

        imgui.end()

    def draw(dt):
        # gl.glDisable(gl.GL_TEXTURE_2D)
        # gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        # gl.glBegin(gl.GL_QUADS)
        # gl.glVertex3f(-1, -1, -1)
        # gl.glVertex3f(5.0, 0.0, 0.0)
        # gl.glVertex3f(5.0, 5.0, 0.0)
        # gl.glVertex3f(0.0, 5.0, 0.0)
        # gl.glEnd()

        update(dt)
        window.clear()

        imgui.render()

        impl.render(imgui.get_draw_data())

    pyglet.clock.schedule_interval(draw, 1 / 60.0)
    pyglet.app.run()
    impl.shutdown()


if __name__ == "__main__":
    main()
