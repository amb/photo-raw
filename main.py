import os
import cv2
import exifread
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from OpenImageIO import ImageOutput, ImageSpec

print(rawpy.flags)
# from scipy import interpolate, ndimage

from image_cl import CLDev, oklab
from image_cl import pycl as cl

# Init cl_builder using CL device 0
cl_builder = CLDev(0)


def bilateral(pix, radius, preserve):
    src = f"""
    #define READP(x,y) read_imagef(input, sampler, (int2)(gx, gy))
    #define POW2(a) ((a) * (a))
    kernel void bilateral(
        const int width,
        const int height,
        const float radius,
        const float preserve,
        __read_only image2d_t input,
        __write_only image2d_t output)
    {{
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const int2 loc = (int2)(gx, gy);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

        int n_radius = ceil(radius);
        int u, v;
        float4 center_pix  = READP(input, loc);
        float4 accumulated = 0.0f;
        float4 tempf       = 0.0f;
        float  count       = 0.0f;
        float  diff_map, gaussian_weight, weight;
        for (v = -n_radius;v <= n_radius; ++v) {{
            for (u = -n_radius;u <= n_radius; ++u) {{
                tempf = read_imagef(input, sampler, (int2)(gx + u, gy + v));
                diff_map = exp (
                    - (   POW2(center_pix.x - tempf.x)
                        + POW2(center_pix.y - tempf.y)
                        + POW2(center_pix.z - tempf.z))
                    * preserve);
                gaussian_weight = exp( - 0.5f * (POW2(u) + POW2(v)) / radius);
                weight = diff_map * gaussian_weight;
                accumulated += tempf * weight;
                count += weight;
            }}
        }}
        write_imagef(output, loc, accumulated / count);
    }}"""

    k = cl_builder.build(
        "bilateral", src, (cl.cl_int, cl.cl_int, cl.cl_float, cl.cl_float, cl.cl_image, cl.cl_image)
    )
    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_builder.run(k, [radius, preserve], [img.data], [out.data], shape=(img.width, img.height))
    return out.to_numpy()


def defish_cl(pix):
    src = f"""
    #define READP(x,y) read_imagef(input, sampler, (int2)(gx, gy))
    kernel void defish(
        const int width,
        const int height,
        __read_only image2d_t input,
        __write_only image2d_t output)
    {{
        const float zoom = 0.55;
        const float twist = 3.8;

        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const int2 loc = (int2)(gx, gy);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP |
            CLK_FILTER_LINEAR;

        const float distx = (float)(gx-width/2);
        const float disty = (float)(gy-height/2);

        float r = sqrt(distx*distx + disty*disty);
        r = r == 0.0 ? 1.0 : r;
        float theta = r/(float)(width);

        float nr = atan(theta * zoom * twist) * 1.273239 * (float)(width) * pow(1.0 / twist, 0.5);
        float dx = distx * nr / r + (float)(width/2);
        float dy = disty * nr / r + (float)(height/2);

        float4 col = read_imagef(input, sampler, (float2)(dx, dy));
        write_imagef(output, loc, col);
    }}"""

    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)
    k = cl_builder.build("defish", src, (cl.cl_int, cl.cl_int, cl.cl_image, cl.cl_image))
    cl_builder.run(k, [], [img.data], [out.data], shape=(img.width, img.height))
    return out.to_numpy()


def clear_cl(cl_img):
    src = """
    kernel void clear(
        const int width,
        const int height,
        __write_only image2d_t output)
    {
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const int2 loc = (int2)(gx, gy);

        write_imagef(output, loc, (float4)(0.0));
    }"""

    k = cl_builder.build("clear", src, (cl.cl_int, cl.cl_int, cl.cl_image))
    cl_builder.run(k, [], [], [cl_img.data], shape=(cl_img.width, cl_img.height))


def create_rot_matrix(x, y, rot):

    m_y = y * np.pi
    m_x = x * np.pi * 2.0
    m_r = rot * np.pi

    # TODO: y=depth? why?
    # x=right, z=up

    cosb, sinb = np.cos(m_r), np.sin(m_r)
    cosy, siny = np.cos(m_y), np.sin(m_y)
    cosa, sina = np.cos(m_x), np.sin(m_x)

    # around z-axis (x-angle)
    i_rza = np.array(
        [
            [cosa, -sina, 0.0],
            [sina, cosa, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # around y-axis (roll)
    i_rya = np.array(
        [
            [cosb, 0.0, sinb],
            [0.0, 1.0, 0.0],
            [-sinb, 0.0, cosb],
        ],
        dtype=np.float32,
    )

    # around x-axis (y-angle)
    i_rxa = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosy, -siny],
            [0.0, siny, cosy],
        ],
        dtype=np.float32,
    )

    return i_rxa @ i_rya @ i_rza


def stereographic_to_equirect(pix, out, offset_x, offset_y, optic_x, optic_y, rot, fov):
    src = f"""
    kernel void stereographic_to_equirect(
        const int width,
        const int height,
        const float optic_x,
        const float optic_y,
        const float m_fov,
        const __global float* i_mtx,
        __read_only image2d_t input,
        __read_write image2d_t output)
    {{
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const float2 gloc = (float2)(gx, gy);
        const int2 iloc = (int2)(gx, gy);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP |
            CLK_FILTER_LINEAR;

        const float fwidth = (float)(width);
        const float fheight = (float)(height);

        const float EWIDTH = 2048.0;
        const float EHEIGHT = 1024.0;

        const float map_x = (float)(gx)*fwidth/EWIDTH;
        const float map_y = (float)(gy)*fheight/EHEIGHT;

        const float FOV = M_PI * m_fov / 180.0;

        // Calculate vector in equirectangular surface
        float theta = 2.0 * M_PI * (map_x / fwidth - 0.5); // -PI .. PI
        float phi = M_PI * (map_y / fheight - 0.5); // -PI/2 .. PI/2
        float px = cos(phi) * sin(theta);
        float py = cos(phi) * cos(theta);
        float pz = sin(phi);

        // Apply matrix to transform view vectors
        float pxn = px * i_mtx[0] + py * i_mtx[1] + pz * i_mtx[2];
        float pyn = px * i_mtx[3] + py * i_mtx[4] + pz * i_mtx[5];
        float pzn = px * i_mtx[6] + py * i_mtx[7] + pz * i_mtx[8];

        px = pxn;
        py = pyn;
        pz = pzn;

        // Calculate equirectangular angle and radius
        theta = atan2(pz, px); // angle of vector to y-plane
        phi = atan2(sqrt(px*px + pz*pz), py); // angle of vector to x-plane

        // https://en.wikipedia.org/wiki/Fisheye_lens#Mapping_function
        // float r = fwidth * phi / FOV;
        float r = 2.0 * fwidth * tan(phi * 0.5 / FOV);

        // Pixel in fisheye space
        float dx = (0.5 + optic_x) * fwidth + r * cos(theta);
        float dy = (0.5 + optic_y) * fheight + r * sin(theta);

        // Write out
        float4 col;
        if (dx > 0.0 && dy > 0.0 && dx < fwidth && dy < fheight) {{
            col = read_imagef(input, sampler, (float2)(dx, dy));

            // TODO: should probably be r/(len(width, height))
            float tr = 1.0 - r / fwidth;

            // Set alpha to distance from origin
            if (tr > 0.0) {{
                // Vignetting correction for Samyang 7.5mm
                float vg = 1.0;
                vg += -0.2271550 *pow(tr, 2);
                vg += -0.1040244 *pow(tr, 3);
                vg +=  0.0864606 *pow(tr, 4);
                vg += -0.3646185 *pow(tr, 5);
                vg +=  0.1749058 *pow(tr, 6);
                col *= 1.0f/vg;
                col.w = sqrt(tr);
            }}

            if (col.w > read_imagef(output, iloc).w)
                write_imagef(output, iloc, col);
        }}

    }}"""

    k = cl_builder.build(
        "stereographic_to_equirect",
        src,
        (
            cl.cl_int,
            cl.cl_int,
            cl.cl_float,
            cl.cl_float,
            cl.cl_float,
            cl.cl_mem,
            cl.cl_image,
            cl.cl_image,
        ),
    )
    img = cl_builder.new_image_from_ndarray(pix)
    # print(img.width, img.height)
    # print(out.width, out.height)
    cl_mtx = cl_builder.to_buffer(create_rot_matrix(offset_x, offset_y, rot).flatten())
    cl_builder.run(
        k,
        [optic_x, optic_y, fov, cl_mtx],
        [img.data],
        [out.data],
        input_shape=(img.width, img.height),
        shape=(out.width, out.height),
    )
    # return out.to_numpy()


def plot_image_array(rgb):
    # Show image with Matplotlib
    divider = rgb.shape[1] / 14
    plt.rcParams["figure.figsize"] = [rgb.shape[1] / divider, rgb.shape[0] / divider]
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # plt.imshow(rgb[::4, ::4, :])
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()


def save_image_as_exr(rgb, path):
    assert len(rgb.shape) == 3
    assert rgb.shape[2] == 4, "Input needs to be RGBA"
    assert rgb.dtype == np.float32, "Input needs to be np.float32"

    # filename = f"developed/{os.path.basename(path)}.{save_as.lower()}"
    filename = f"developed/{os.path.basename(path)}.exr"
    spec = ImageSpec(rgb.shape[1], rgb.shape[0], rgb.shape[2], "float")
    out = ImageOutput.create(filename)
    assert out, f"Error: Unable to open OIIO EXR file for writing"
    out.open(filename, spec)
    out.write_image(rgb)
    out.close()


def show_exif(path):
    # Print some EXIF data
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=True)
        # print("\nEXIF data:")
        # print(
        #     f'Width: {tags["Image ImageWidth"].values}, Height: '
        #     f'{tags["Image ImageLength"].values}, BPS: {tags["Image BitsPerSample"].values}'
        # )
        # print(f'Make: {tags["Image Make"].values}, Model: {tags["Image Model"].values}')
        # print(
        #     f'Orientation: {tags["Image Orientation"].values}, '
        #     f'Software: {tags["Image Software"].values}'
        # )
        # print(f'Date: {tags["Image DateTime"].values},
        # Copyright: {tags["Image Copyright"].values}')
        for key, value in tags.items():
            # kl = key.lower()
            print(f"{key}: {repr(value)}")
            # do not print (uninteresting) binary thumbnail data
            # if "Image" in key or "GPS" in key: # or "EXIF" in key:
            #     print(f"{key}: {repr(value)}")
            # elif "MakerNote":
            #     print(f"{key}: {repr(value)} {type(value)}")


def show_raw_data(raw):
    print(f"Raw type:                     {raw.raw_type}")
    print(f"Number of colors:             {raw.num_colors}")
    print(f"Color description:            {raw.color_desc}")
    print(f"Raw pattern:                  {raw.raw_pattern.tolist()}")
    print(f"Black levels:                 {raw.black_level_per_channel}")
    print(f"White level:                  {raw.white_level}")
    print(f"Color matrix:                 {raw.color_matrix.tolist()}")
    print(f"XYZ to RGB conversion matrix: {raw.rgb_xyz_matrix.tolist()}")
    print(f"Camera white balance:         {raw.camera_whitebalance}")
    print(f"Daylight white balance:       {raw.daylight_whitebalance}")

    # print("Rawpy supported demosaic algos:")
    # for i in range(13):
    #     t = rawpy.DemosaicAlgorithm(i)
    #     if t.isSupported:
    #         print(t)
    # print()


def process_raw(path, denoise=True):
    raw = rawpy.imread(path)
    rgb = raw.postprocess(
        gamma=(1, 1),
        # exp_shift=3,
        no_auto_bright=True,
        use_camera_wb=True,
        output_bps=16,
        # fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(2),
        # dcb_iterations=4,
        # demosaic_algorithm=rawpy.DemosaicAlgorithm(4),
    )

    rgb = np.float32(rgb) / (2 ** 16 - 1)

    # gamma compression, exposure correction
    exposure_steps = 0.0
    rgb *= 2 ** exposure_steps
    rgb = oklab.linear_to_srgb(rgb, clamp=True)

    # contrasty curves (I have no idea what i'm doing here, didn't look this up)
    # rgb = np.abs(rgb * 2 - 1) ** 0.8 * np.sign(rgb * 2 - 1) * 0.5 + 0.5
    # rgb = rgb * 0.5 + ((np.sin((rgb - 0.5) * np.pi) + 1.0) * 0.5) * 0.5

    rgb4 = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=rgb.dtype)
    rgb4[..., :3] = rgb
    rgb4[..., 3] = 1.0

    if denoise:
        print("Denoising...")

        lab = oklab.srgb_to_Lab(rgb4)
        L = lab[..., 0]
        # lab = bilateral(lab, 64.0, 256.0)
        lab = bilateral(lab, 64.0, 128.0)
        # Only chroma
        lab[..., 0] = L
        # lab = bilateral(lab, 16.0, 512.0)
        rgb4 = oklab.Lab_to_srgb(lab)

        # TODO: is there 16-bit denoising?
        rgb_i = cv2.fastNlMeansDenoisingColored(np.uint8(rgb4[..., :3] * 255), None, 5, 5, 7, 15)
        rgb4[..., :3] = np.float32(rgb_i / 255.0)

    return rgb4, raw


def get_tilt(filename):
    """use the great exiftool.exe to parse out roll and pitch angle"""
    import subprocess

    out = subprocess.run(["exiftool", "-RollAngle", "-PitchAngle", filename], capture_output=True)
    lines = out.stdout.decode("ASCII").split("\n")
    if len(lines) != 3:
        return None, None
    lines = lines[:2]
    roll_angle = float(lines[0].split(":")[-1])
    pitch_angle = float(lines[1].split(":")[-1])

    return roll_angle, pitch_angle


# Approximate values, don't take it as a science
# Sun: 100,000 lux
# Daylight: 10000 lux
# Overcast day/tv studio: 1000 lux
# Office: 400 lux
# Livingroom: 50 lux
# Full moon: 0.2 lux
# Sun multiplier: 50x = ~5.7 steps

# OM-D E-M10 II: 12.0 Ev dynamic range w/ 200 ISO
# Load and convert raw camera input from a fisheye lens

# show_raw_data(raw)
# show_exif(raws[i])

import glob

image_arrays = []
rotations = []
# raws = glob.glob("D:/photo/2021.7/pano2/*.orf")
# raws = glob.glob("D:/swap/downloads/*.orf")
# raws = glob.glob("test/P7240158.orf")
raws = glob.glob("raws4/*.orf")
raws = raws[:4]
# raws = raws[:1] + raws[-1:]
# raws = [raws[0]]
print("Total raws in folder:", len(raws))

# image, raw = process_raw(raws[1], denoise=False)


if True:
    out_cl = cl_builder.new_image(2048, 1024)
    for i, filename in enumerate(raws):
        print(f"Loading: {filename} ({os.path.basename(filename)})")
        roll_angle, pitch_angle = get_tilt(filename)
        if roll_angle and pitch_angle:
            # Only add images with angles
            print(f"Roll: {roll_angle}, Pitch: {pitch_angle}")
            image, raw_pixels = process_raw(filename, denoise=False)
            image_arrays.append(image)
            rotations.append((roll_angle, pitch_angle))

    print("Plotting final...")

    def new_image_data(width, height, params):
        # final = np.zeros((512, 1024, 4), dtype=np.float32)
        clear_cl(out_cl)
        for img_i, img in enumerate(image_arrays):
            stereographic_to_equirect(
                img,
                out_cl,
                params["x_locs"][img_i],
                params["y_locs"][img_i],
                params["x_optic"][img_i],
                params["y_optic"][img_i],
                params["rot"][img_i],
                params["fov"],
            )
        final = out_cl.to_numpy()[::2, ::2, :]
        # normalize
        final -= np.min(final)
        final /= np.max(final)
        final[..., 3] = 1.0
        return (final * 255.0).astype("uint8")

    import gui

    # samyang fisheye (theorethical): 7.5mm
    # -0.072 rad distort, -66.6 image hcenter shift, 40.0 vcenter shift
    # fisheye factor: -0.488398
    # shift: long=-0.023%, short=0.23%
    # distortion (a, b, c): -0.045, 0.14, -0.11
    # m4/3: (18mm w, 13.5mm h, 22.5mm d) imaging area: (17.3mm w, 13.0mm h, 21.6mm d) crop: 2.0
    focal_distance = 7.5
    sensor_diagonal = 21.6
    sensor_horizontal = 17.3

    gui.generate_image_data = new_image_data
    gui.main(
        [i / len(image_arrays) for i in range(len(image_arrays))],
        # v[0]=roll, v[1]=pitch
        [v[1] / 180 for v in rotations],
        [v[0] / 180 for v in rotations],
        np.arctan(sensor_horizontal / 2 / focal_distance) * 180.0 / np.pi * 2.0,
        # 144.0,
    )
