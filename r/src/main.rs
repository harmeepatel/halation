use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, Pixel, Rgb, RgbImage,
    imageops::FilterType,
};
use imageproc::{
    contrast::{self, ThresholdType},
    distance_transform::Norm,
    edges, filter, morphology,
};
use minifb::{Window, WindowOptions};
use std::path::Path;

const IMAGE_PATH: &str = "./../imgs/park.png";

const WIDTH: u32 = 1440;
const HEIGHT: u32 = 1080;

const BLOOM_THRESHOLD: u8 = 80;
const BLOOM_THRESHOLD_MIN: f32 = 0.8;
const BLOOM_THRESHOLD_MAX: f32 = 1.0;
const BLOOM_RADIUS: f32 = 8.0;
const BLOOM_INTENSITY: f32 = 0.3;

const DIFFUSION_THRESHOLD: u8 = 245;
const DIFFUSION_RADIUS: f32 = 8.0;
const DIFFUSION_INTENSITY: f32 = 0.8;

const BLOOM_PASSES: &[(u32, f32, f32)] = &[
    (1, 6.0, 6.0),  // Tighter, more intense core bloom
    (1, 12.0, 2.0),  // Wider, medium spread
    (1, 18.0, 1.0), // Very wide, subtle halo
    (1, 22.0, 0.8), // Very wide, subtle halo
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img: DynamicImage = match image::open(&Path::new(IMAGE_PATH)) {
        Ok(image) => {
            println!("Image loaded successfully!");
            image
        }
        Err(e) => {
            eprintln!("Error loading image: {}", e);
            eprintln!(
                "Please ensure '{}' exists and is a valid image file.",
                IMAGE_PATH
            );
            eprintln!("Using a blank window as a fallback.");
            // Create a dummy image for fallback if loading fails
            DynamicImage::new_rgb8(WIDTH as u32, HEIGHT as u32)
        }
    };

    let (w, h) = if img.dimensions().0 as f32 / 2.8 > WIDTH as f32 {
        (
            (img.dimensions().0 as f32 / 2.8) as u32,
            (img.dimensions().1 as f32 / 2.8) as u32,
        )
    } else {
        ((img.dimensions().0 as f32 / 4.2) as u32, (img.dimensions().1 as f32 / 4.2) as u32)
    };

    let img_resize = image::imageops::resize(
        &img.to_rgb8(),
        w,
        h,
        FilterType::Gaussian, // High quality resampling
    );
    let img_resize = image::DynamicImage::ImageRgb8(img_resize);
    let (img_width, img_height) = img_resize.dimensions();
    println!("Image dimensions: {}x{}", img_width, img_height);

    let img_bloom = add_bloom(&img_resize.clone());

    let img_threshold = bloom(&img_resize.to_luma8(), BLOOM_THRESHOLD, BLOOM_RADIUS);
    let mut buf_original: Vec<u32> = Vec::new();
    write_buffer(&mut buf_original, &img_resize);

    let mut buf_bloom: Vec<u32> = Vec::new();
    write_buffer(&mut buf_bloom, &img_bloom);

    let mut buf_threshold: Vec<u32> = Vec::new();
    write_buffer(&mut buf_threshold, &img_threshold.into());

    // ----------------------------------------------------------------------
    let mut win_bloom = Window::new(
        "bloomed",
        img_width as usize,
        img_height as usize,
        WindowOptions::default(),
    )?;
    win_bloom.set_target_fps(60);
    let mut win_original = Window::new(
        "original",
        img_width as usize,
        img_height as usize,
        WindowOptions::default(),
    )?;
    win_original.set_target_fps(60);
    let mut win_threshold = Window::new(
        "threshold",
        img_width as usize,
        img_height as usize,
        WindowOptions::default(),
    )?;
    win_threshold.set_target_fps(60);

    while win_bloom.is_open(){
        win_threshold
            .update_with_buffer(&buf_threshold, img_width as usize, img_height as usize)
            .unwrap_or_else(|e| {
                eprintln!("Error updating window: {}", e); });
        win_original
            .update_with_buffer(&buf_original, img_width as usize, img_height as usize)
            .unwrap_or_else(|e| {
                eprintln!("Error updating window: {}", e);
            });
        win_bloom
            .update_with_buffer(&buf_bloom, img_width as usize, img_height as usize)
            .unwrap_or_else(|e| {
                eprintln!("Error updating window: {}", e);
            });
    }

    println!("Window closed. Exiting.");
    Ok(())
}

fn write_buffer(buf: &mut Vec<u32>, img: &DynamicImage) {
    let img = img.to_rgba8();
    for pixel in img.pixels() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        let a = pixel[3] as u32;
        buf.push((a << 24) | (r << 16) | (g << 8) | b);
    }
}

fn add_diffusion(og: &DynamicImage) -> DynamicImage {
    let (img_w, img_h) = og.clone().into_rgb8().dimensions();
    let og_rgb = og.clone().into_rgb8();
    let bloomer_luma = og.clone().into_luma8();

    let thresholded_luma =
        contrast::threshold(&bloomer_luma, DIFFUSION_THRESHOLD, ThresholdType::ToZero);
    let blurred_glow = filter::gaussian_blur_f32(&thresholded_luma, DIFFUSION_RADIUS);

    let buf = ImageBuffer::from_fn(img_w, img_h, |x, y| {
        let original_pixel = og_rgb.get_pixel(x, y).to_rgb().0;
        let glow_pixel_luma = blurred_glow.get_pixel(x, y).to_luma()[0] as f32;

        let r_orig = original_pixel[0] as f32;
        let g_orig = original_pixel[1] as f32;
        let b_orig = original_pixel[2] as f32;

        let r_new = (r_orig + (glow_pixel_luma * DIFFUSION_INTENSITY)).min(255.0);
        let g_new = (g_orig + (glow_pixel_luma * DIFFUSION_INTENSITY)).min(255.0);
        let b_new = (b_orig + (glow_pixel_luma * DIFFUSION_INTENSITY)).min(255.0);

        image::Rgb([r_new as u8, g_new as u8, b_new as u8])
    });

    DynamicImage::ImageRgb8(buf)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn bloom(img: &GrayImage, threshold: u8, radius: f32) -> DynamicImage {
    let thresh = contrast::threshold(&img, threshold, ThresholdType::Binary);
    let glow = edges::canny(&thresh, 50.0, 150.0);
    let dilated = morphology::dilate(&glow, Norm::L1, 5);
    let blurred = filter::gaussian_blur_f32(&dilated, radius);
    DynamicImage::ImageLuma8(blurred)
}

fn add_bloom(og: &DynamicImage) -> DynamicImage {
    // let (img_w, img_h) = og.clone().into_rgb8().dimensions();
    // let og_rgb = og.clone().into_rgb8();
    // let bloomer_luma = og.clone().into_luma8();
    //
    // let blurred_glow = bloom(&bloomer_luma, BLOOM_THRESHOLD, BLOOM_RADIUS);
    //
    // // let thresholded_luma = threshold(&bloomer_luma, BLOOM_THRESHOLD, ThresholdType::Binary);
    // // let edge_glow = edges::canny(&thresholded_luma, 50.0, 150.0);
    // // let blurred_glow = filter::gaussian_blur_f32(&edge_glow, BLOOM_RADIUS);
    //
    // let buf = ImageBuffer::from_fn(img_w, img_h, |x, y| {
    //     let original_pixel = og_rgb.get_pixel(x, y).to_rgb().0;
    //     let glow_pixel_luma = blurred_glow.get_pixel(x, y).to_luma()[0] as f32;
    //
    //     let r_orig = original_pixel[0] as f32;
    //     let g_orig = original_pixel[1] as f32;
    //     let b_orig = original_pixel[2] as f32;
    //
    //     let r_new = (r_orig + (glow_pixel_luma * BLOOM_INTENSITY)).min(255.0);
    //     let g_new = (g_orig + (glow_pixel_luma * BLOOM_INTENSITY)).min(255.0);
    //     let b_new = (b_orig + (glow_pixel_luma * BLOOM_INTENSITY)).min(255.0);
    //
    //     image::Rgb([r_new as u8, g_new as u8, b_new as u8])
    // });
    //
    // DynamicImage::ImageRgb8(buf)

    let (img_w, img_h) = og.dimensions();

    let original_rgb_f32: ImageBuffer<Rgb<f32>, Vec<f32>> =
        ImageBuffer::from_fn(img_w, img_h, |x, y| {
            let p = og.get_pixel(x, y).to_rgb();
            Rgb([
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ])
        });

    let mut final_bloom_layer: ImageBuffer<Rgb<f32>, Vec<f32>> =
        ImageBuffer::from_pixel(img_w, img_h, Rgb([0.0, 0.0, 0.0]));

    for &(downsample_factor, blur_sigma, pass_multiplier) in BLOOM_PASSES.iter() {
        let current_w = img_w / downsample_factor;
        let current_h = img_h / downsample_factor;

        let downsampled_original = image::imageops::resize(
            &original_rgb_f32,
            current_w,
            current_h,
            FilterType::Triangle, // Good for downsampling
        );

        let current_bloom_mask = ImageBuffer::from_fn(current_w, current_h, |x, y| {
            let pixel = downsampled_original.get_pixel(x, y);
            let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];

            let bloom_contribution =
                smoothstep(BLOOM_THRESHOLD_MIN, BLOOM_THRESHOLD_MAX, luminance);

            let r_contrib = pixel[0] * bloom_contribution;
            let g_contrib = pixel[1] * bloom_contribution;
            let b_contrib = pixel[2] * bloom_contribution;

            image::Rgb([r_contrib, g_contrib, b_contrib])
        });

        let blurred_pass = filter::gaussian_blur_f32(&current_bloom_mask, blur_sigma);
        let upsampled_blurred_pass = image::imageops::resize(
            &blurred_pass,
            img_w,
            img_h,
            FilterType::CatmullRom, // Good for upsampling
        );

        for y in 0..img_h {
            for x in 0..img_w {
                let current_final_pixel = final_bloom_layer.get_pixel(x, y);
                let upsampled_pixel = upsampled_blurred_pass.get_pixel(x, y);

                let r_acc = current_final_pixel[0] + upsampled_pixel[0] * pass_multiplier;
                let g_acc = current_final_pixel[1] + upsampled_pixel[1] * pass_multiplier;
                let b_acc = current_final_pixel[2] + upsampled_pixel[2] * pass_multiplier;

                final_bloom_layer.put_pixel(x, y, Rgb([r_acc, g_acc, b_acc]));
            }
        }
    }

    let final_image_f32 = ImageBuffer::from_fn(img_w, img_h, |x, y| {
        let original_pixel = original_rgb_f32.get_pixel(x, y);
        let bloom_pixel = final_bloom_layer.get_pixel(x, y);

        let r_final = (original_pixel[0] + bloom_pixel[0] * BLOOM_INTENSITY).min(1.0);
        let g_final = (original_pixel[1] + bloom_pixel[1] * BLOOM_INTENSITY).min(1.0);
        let b_final = (original_pixel[2] + bloom_pixel[2] * BLOOM_INTENSITY).min(1.0);

        Rgb([r_final, g_final, b_final])
    });

    let final_image_u8: RgbImage = ImageBuffer::from_fn(img_w, img_h, |x, y| {
        let p = final_image_f32.get_pixel(x, y);
        Rgb([
            (p[0] * 255.0) as u8,
            (p[1] * 255.0) as u8,
            (p[2] * 255.0) as u8,
        ])
    });

    DynamicImage::ImageRgb8(final_image_u8)
}
