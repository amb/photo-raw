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


def fish_to_equirect(pix):
    src = f"""
    kernel void fish_to_equirect(
        const int width,
        const int height,
        __read_only image2d_t input,
        __write_only image2d_t output)
    {{
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const float2 gloc = (float2)(gx, gy);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP |
            CLK_FILTER_LINEAR;

        const float fwidth = (float)(width);
        const float fheight = (float)(height);

        const float up_angle = 30.0 * M_PI / 360.0;

        const float EWIDTH = 2048.0;
        const float EHEIGHT = 1024.0;

        const float map_x = (float)(gx*width)/EWIDTH;
        const float map_y = (float)(gy*height)/EHEIGHT;

        // Samyang 7.5mm f/3.5 = 99.1 Hfov
        // -0.072 rad distort, -66.6 image hcenter shift, 40.0 vcenter shift
        // ~30 pitch, -2 roll
        const float FOV = M_PI * 99.1 / 180.0;

        // Polar angles
        float theta = 2.0 * M_PI * (map_x / fwidth - 0.5); // -pi to pi
        float phi = M_PI * (map_y / fheight - 0.5); // -pi/2 to pi/2

        // Vector in 3D space
        float px = cos(phi) * sin(theta);
        float py = cos(phi) * cos(theta);
        float pz = sin(phi);

        // Calculate fisheye angle and radius
        theta = atan2(pz, px);
        phi = atan2(sqrt(px*px + pz*pz), py);
        float r = fwidth * phi / FOV;

        // Pixel in fisheye space
        const float offset_x = 0.0;
        // const float offset_y = 0.0;
        const float offset_y = up_angle;

        float dx = (0.5 + offset_x) * fwidth + r * cos(theta);
        float dy = (0.5 + offset_y) * fheight + r * sin(theta);

        // Write out
        float4 col = read_imagef(input, sampler, (float2)(dx, dy));
        write_imagef(output, (int2)(gx, gy), col);
    }}"""

    k = cl_builder.build("fish_to_equirect", src, (cl.cl_int, cl.cl_int, cl.cl_image, cl.cl_image))
    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(2048, 1024)
    cl_builder.run(
        k, [], [img.data], [out.data], input_shape=(img.width, img.height), shape=(2048, 1024)
    )
    return out.to_numpy()


def stereographic_to_equirect(pix, out, offset_x, offset_y):
    src = f"""
    kernel void stereographic_to_equirect(
        const int width,
        const int height,
        const float offset_x,
        const float offset_y,
        __read_only image2d_t input,
        __write_only image2d_t output)
    {{
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const float2 gloc = (float2)(gx, gy);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP |
            CLK_FILTER_LINEAR;

        const float fwidth = (float)(width);
        const float fheight = (float)(height);

        const float up_angle = offset_y * M_PI / 180.0;

        const float EWIDTH = 2048.0;
        const float EHEIGHT = 1024.0;

        const float map_x = (float)(gx)*fwidth/EWIDTH;
        const float map_y = (float)(gy)*fheight/EHEIGHT;

        // Samyang 7.5mm f/3.5 on MFT, 2x crop = 99.1 Hfov calculated (ideal=100.4, 77.3)
        // -0.072 rad distort, -66.6 image hcenter shift, 40.0 vcenter shift
        // ~30 pitch, -2 roll
        // const float FOV = M_PI * 99.1 / 180.0;
        // const float FOV = M_PI * 160.0 / 180.0;
        const float FOV = M_PI * 215.0 / 180.0;

        // Calculate vector in equirectangular surface
        float theta = 2.0 * M_PI * (offset_x + map_x / fwidth - 0.5); // -PI .. PI
        float phi = M_PI * (map_y / fheight - 0.5); // -PI/2 .. PI/2
        float px = cos(phi) * sin(theta);
        float py = cos(phi) * cos(theta);
        float pz = sin(phi);

        // Rotate around x-axis
        float pyn = py * cos(up_angle) - pz * sin(up_angle);
        float pzn = py * sin(up_angle) + pz * cos(up_angle);
        py = pyn;
        pz = pzn;

        // Calculate equirectangular angle and radius
        theta = atan2(pz, px); // angle of vector to y-plane
        phi = atan2(sqrt(px*px + pz*pz), py); // angle of vector to x-plane

        // https://en.wikipedia.org/wiki/Fisheye_lens#Mapping_function
        float r = fwidth * phi / FOV;
        // float r = 2.0 * fwidth * tan(phi * 0.5 / FOV);

        // Pixel in fisheye space
        float dx = (0.5) * fwidth + r * cos(theta);
        float dy = (0.5) * fheight + r * sin(theta);

        // Write out
        float4 col = read_imagef(input, sampler, (float2)(dx, dy));
        
        // Set alpha to distance from origin
        float tr = 1.0 - 2.0 * r / fwidth;
        if (tr > 0.0) {{
            col.w = sqrt(tr);
        }} else {{
            col.w = 0.0;
        }}
        //if (col.w < 0.0)
        //    col.w = 0.0;

        if (dx > 0.0 && dy > 0.0 && dx < fwidth && dy < fheight) {{
            write_imagef(output, (int2)(gx, gy), col);
        }}
    }}"""

    k = cl_builder.build(
        "stereographic_to_equirect",
        src,
        (cl.cl_int, cl.cl_int, cl.cl_float, cl.cl_float, cl.cl_image, cl.cl_image),
    )
    img = cl_builder.new_image_from_ndarray(pix)
    cl_builder.run(
        k,
        [offset_x, offset_y],
        [img.data],
        [out.data],
        input_shape=(img.width, img.height),
        shape=(2048, 1024),
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

    print("RAW postprocess...")
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
# raws = glob.glob("D:/photo/2021.7/pano2/*.orf")
# raws = glob.glob("D:/swap/downloads/*.orf")
raws = glob.glob("raws2/*.orf")
print("Total raws in folder:", len(raws))

print("Using Exiftool to extract camera tilt...")
roll_angle, pitch_angle = get_tilt(raws[0])
print("Roll angle:", roll_angle)
print("Pitch angle:", pitch_angle)


# image, raw = process_raw(raws[1], denoise=False)

if False:
    out = cl_builder.new_image(2048, 1024)
    x_locs = [0.0, 0.502]
    final = np.zeros((1024, 2048, 4), dtype=np.float32)
    for i, filename in enumerate(raws[:2]):
        print(f"Loading: {filename} ({os.path.basename(filename)})")
        image, raw_pixels = process_raw(filename, denoise=False)
        stereographic_to_equirect(image, out, x_locs[i], -5.0)
        # image_arrays.append(image)
        np_image = out.to_numpy()
        locs = np_image[..., 3] > final[..., 3]
        final[locs] = np_image[locs]
        # final = final + np_image
    print("Plotting final...")
    final[..., 3] = 1.0
    plot_image_array(final)
