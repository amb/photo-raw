import cv2
import exifread
import matplotlib.pyplot as plt
import numpy as np
import OpenImageIO as oiio
import rawpy
from OpenImageIO import ImageBuf, ImageOutput, ImageSpec

# from scipy import interpolate, ndimage

from image_cl import CLDev
from image_cl import pycl as cl
from image_cl import oklab

cl_builder = CLDev(0)


# Load and convert raw camera input from a fisheye lens
# path = "raws/P1010017.ORF"
path = "test/P1010047.ORF"
print("Loading:", path)

# Print some EXIF data
if True:
    with open(path, "rb") as f:
        tags = exifread.process_file(f)
        print("\nEXIF data:")
        print(
            f'Width: {tags["Image ImageWidth"].values}, Height: '
            f'{tags["Image ImageLength"].values}, BPS: {tags["Image BitsPerSample"].values}'
        )
        print(f'Make: {tags["Image Make"].values}, Model: {tags["Image Model"].values}')
        print(
            f'Orientation: {tags["Image Orientation"].values}, '
            f'Software: {tags["Image Software"].values}'
        )
        print(f'Date: {tags["Image DateTime"].values}, Copyright: {tags["Image Copyright"].values}')
        print()
        # for key, value in tags.items():
        #     # do not print (uninteresting) binary thumbnail data
        #     if "Image" in key or "GPS" in key: # or "EXIF" in key:
        #         print(f"{key}: {repr(value)}")
        #     # elif "MakerNote":
        #     #     print(f"{key}: {repr(value)} {type(value)}")

raw = rawpy.imread(path)

print(f"Raw type:                     {raw.raw_type}")
print(f"Number of colors:             {raw.num_colors}")
print(f"Color description:            {raw.color_desc}")
print(f"Raw pattern:                  {raw.raw_pattern.tolist()}")
print(f"Black levels:                 {raw.black_level_per_channel}")
print(f"White level:                  {raw.white_level}")
# print(f'Color matrix:                 {raw.color_matrix.tolist()}')
# print(f'XYZ to RGB conversion matrix: {raw.rgb_xyz_matrix.tolist()}')
print(f"Camera white balance:         {raw.camera_whitebalance}")
print(f"Daylight white balance:       {raw.daylight_whitebalance}")

rgb = raw.postprocess(
    gamma=(1, 1), exp_shift=3, no_auto_bright=True, use_camera_wb=True, output_bps=16
)
print(rgb.shape, rgb.dtype)
height = rgb.shape[0]
width = rgb.shape[1]
rgb = np.float32(rgb) / (2 ** 16 - 1)

# gamma compression
rgb = 1.0 * rgb ** 0.6


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


print("Denoising...")

rgb4 = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=rgb.dtype)
rgb4[..., :3] = rgb
rgb4[..., 3] = 1.0
lab = oklab.srgb_to_Lab(rgb4)
L = lab[..., 0]
# lab = bilateral(lab, 64.0, 256.0)
lab = bilateral(lab, 32.0, 128.0)
lab[..., 0] = L
lab = bilateral(lab, 16.0, 512.0)
rgb = oklab.Lab_to_srgb(lab)[..., :3]

# TODO: is there 16-bit denoising?
# rgb_i = cv2.fastNlMeansDenoisingColored(np.uint8(rgb * 255), None, 15, 20, 7, 15)
# rgb = np.float32(rgb_i / 255.0)

# TODO: takes forever
# from bm3d import bm3d_rgb
# rgb = bm3d_rgb(rgb, sigma_est)

print("Fisheye correction...")


def undistort(img, K, D):
    # img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    return cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


# rgb = undistort(
#     rgb,
#     np.array(
#         [
#             [781.3524863867165, 0.0, 794.7118000552183],
#             [0.0, 779.5071163774452, 561.3314451453386],
#             [0.0, 0.0, 1.0],
#         ]
#     ),
#     np.array(
#         [
#             [-0.042595202508066574],
#             [0.031307765215775184],
#             [-0.04104704724832258],
#             [0.015343014605793324],
#         ]
#     ),
# )

# r_src = r_tgt * (1 + k1 * (r_tgt / r_0)^2 + k2 * (r_tgt / r_0)^4)
# k1: 0.16, k2: 0.6
# r_src = r_tgt * (1.0 + 0.16 * (r_tgt / r_0) ** 2.0 + 0.6 * (r_tgt / r_0) ** 4.0)

# where r_0 is halve of the image diagonal and
# r_src and r_tgt are the distances from the focal point
# in the source and target images, respectively.

rgb_source = rgb.copy()


def dist(x, y):
    return np.sqrt(x * x + y * y)


# zoom = 0.8
# zoom = 0.3
zoom = 0.55

distance = np.float32(np.mgrid[:height, :width])
distance[1] -= width / 2.0
distance[0] -= height / 2.0

fc = dist(width * 0.5, height * 0.5)
r = dist(distance[0], distance[1])
r = np.where(r == 0.0, 1.0, r)

theta = r / fc

twist = 3.5
nr = np.arctan(theta * zoom * twist) * 1.273239 * fc * (1.0 / twist) ** 0.5
nr_max = np.max(nr)
print(np.min(theta), np.max(theta))

dx = distance[1] * nr / r
dy = distance[0] * nr / r

# with this, straight things aren't straight anymore, but aspect ratios are correct
# dx = distance[1] * (nr / r) ** 0.5
# dy = distance[0] * (nr / r) ** 0.5

# Limit
dx = np.where(dx > width / 2 - 1.0, 0.0, dx)
dy = np.where(dy > height / 2 - 1.0, 0.0, dy)
dx = np.where(dx < -width / 2 + 1.0, 0.0, dx)
dy = np.where(dy < -height / 2 + 1.0, 0.0, dy)

rgb = rgb_source[np.int32(dy) + height // 2, np.int32(dx) + width // 2, :]
# TODO: write bilinear sampling func (a=(loc-floor(loc)); img=img[iloc]*(1-a)+img[iloc+1]*a; etc)

# Save floating point data as EXR
if False:
    print("Saving...")
    filename = "developed/default.exr"
    spec = ImageSpec(rgb.shape[1], rgb.shape[0], rgb.shape[2], "float")
    # buf = ImageBuf(spec)
    out = ImageOutput.create(filename)
    assert out, "Error: Unable to open OIIO EXR file for writing"
    out.open(filename, spec)
    out.write_image(rgb)
    out.close()

# Show image with Matplotlib
plt.rcParams["figure.figsize"] = [width / 400, height / 400]
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

# plt.imshow(rgb[::4, ::4, :])
plt.imshow(rgb)
plt.axis("off")
plt.show()
