// Add these dependencies to your Cargo.toml:
// eframe = "0.27.2"
// egui = "0.27.2"
// image = "0.25.1" // For image loading and manipulation
// native-dialog = "0.7.0" // For native file dialogs

use eframe::{egui, App, Frame};
use image::{RgbaImage}; // Using RgbaImage for egui compatibility
use std::sync::Arc; // For sharing image data if needed, though not strictly here yet

// Main application struct to hold the state
struct MyApp {
    // Bloom parameters
    bloom_size: f32,
    bloom_intensity: f32,

    // Halation parameters
    halation_size: f32,
    halation_intensity: f32,

    // Original loaded image
    original_image: Option<Arc<RgbaImage>>, // Store as RgbaImage
    // Processed image to display
    processed_image_texture: Option<egui::TextureHandle>,

    // Placeholder for status messages
    status_message: String,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            bloom_size: 0.5,
            bloom_intensity: 0.8,
            halation_size: 10.0,
            halation_intensity: 0.3,
            original_image: None,
            processed_image_texture: None,
            status_message: "Load an image to begin.".to_string(),
        }
    }
}

// Helper function to convert image::RgbaImage to egui::ColorImage
fn to_egui_color_image(img: &RgbaImage) -> egui::ColorImage {
    let size = [img.width() as _, img.height() as _];
    let pixels = img.to_vec();
    egui::ColorImage::from_rgba_unmultiplied(size, &pixels)
}

// Placeholder for your actual image processing logic
fn apply_bloom_and_halation(
    image: &RgbaImage,
    bloom_size: f32,
    bloom_intensity: f32,
    halation_size: f32,
    halation_intensity: f32,
) -> RgbaImage {
    // --- This is where your core image processing will go ---
    // For now, it just clones the image.
    // You'll need to implement:
    // 1. Thresholding to find bright areas.
    // 2. Gaussian blur (or similar) for bloom, controlled by bloom_size.
    // 3. Blending bloom back, controlled by bloom_intensity.
    // 4. Similar steps for halation (e.g., colored blur), controlled by halation_size and halation_intensity.

    println!(
        "> Applying effects with: Bloom(S:{}, I:{}), Halation(S:{}, I:{})",
        bloom_size, bloom_intensity, halation_size, halation_intensity
    );

    // Example: A very simple "intensity" effect for demonstration
    let mut processed = image.clone();
    for pixel in processed.pixels_mut() {
        // A naive intensity boost - replace with real bloom/halation
        pixel.0[0] = (pixel.0[0] as f32 * (1.0 + bloom_intensity * 0.5)).min(255.0) as u8;
        pixel.0[1] = (pixel.0[1] as f32 * (1.0 + bloom_intensity * 0.5)).min(255.0) as u8;
        pixel.0[2] = (pixel.0[2] as f32 * (1.0 + bloom_intensity * 0.5)).min(255.0) as u8;
    }
    processed
    // image.clone() // Replace this with your actual processed image
}


impl App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Image Effect Controls");
            ui.separator();

            // --- File Loading ---
            if ui.button("Load Image").clicked() {
                let path = native_dialog::FileDialog::new()
                    .add_filter("Image Files", &["png", "jpg", "jpeg", "bmp", "gif"])
                    .show_open_single_file();

                match path {
                    Ok(Some(p)) => {
                        match image::open(&p) {
                            Ok(dyn_img) => {
                                let rgba_image = dyn_img.to_rgba8(); // Convert to RGBA8
                                self.original_image = Some(Arc::new(rgba_image));
                                self.status_message = format!("Loaded image: {:?}", p.file_name().unwrap_or_default());
                                // Initially display the original image
                                if let Some(orig_img_arc) = &self.original_image {
                                    let egui_img = to_egui_color_image(orig_img_arc);
                                    self.processed_image_texture = Some(
                                        ctx.load_texture("processed_image", egui_img, Default::default())
                                    );
                                }
                            }
                            Err(e) => {
                                self.status_message = format!("Error loading image: {}", e);
                                eprintln!("Error loading image: {}", e);
                            }
                        }
                    }
                    Ok(None) => { /* User cancelled */ }
                    Err(e) => {
                        self.status_message = format!("File dialog error: {}", e);
                        eprintln!("File dialog error: {}", e);
                    }
                }
            }
            ui.separator();


            // --- Controls Column ---
            egui::SidePanel::left("controls_panel")
                .resizable(true)
                .default_width(250.0)
                .show_inside(ui, |ui| {
                    ui.vertical_centered(|ui| {
                         ui.heading("Parameters");
                    });
                    ui.separator();

                    // --- Bloom Controls ---
                    ui.collapsing("Bloom Effect", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Size:");
                            ui.add(egui::Slider::new(&mut self.bloom_size, 0.0..=50.0).text("px"));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Intensity:");
                            ui.add(egui::Slider::new(&mut self.bloom_intensity, 0.0..=2.0));
                        });
                        ui.label(format!("Current Bloom: Size {:.2}px, Intensity {:.2}", self.bloom_size, self.bloom_intensity));
                    });

                    ui.separator();

                    // --- Halation Controls ---
                    ui.collapsing("Halation Effect", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Size:");
                            ui.add(egui::Slider::new(&mut self.halation_size, 0.0..=100.0).text("px"));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Intensity:");
                            ui.add(egui::Slider::new(&mut self.halation_intensity, 0.0..=1.0));
                        });
                        ui.label(format!("Current Halation: Size {:.2}px, Intensity {:.2}", self.halation_size, self.halation_intensity));
                    });
                    ui.separator();

                     // --- Action Button ---
                    if ui.button("Apply Effects").clicked() {
                        if let Some(orig_img_arc) = &self.original_image {
                            let processed_img_data = apply_bloom_and_halation(
                                orig_img_arc, // Pass a reference to the RgbaImage
                                self.bloom_size,
                                self.bloom_intensity,
                                self.halation_size,
                                self.halation_intensity,
                            );
                            let egui_processed_img = to_egui_color_image(&processed_img_data);

                            // If a texture already exists, update it. Otherwise, create a new one.
                            if let Some(texture_handle) = &mut self.processed_image_texture {
                                texture_handle.set(egui_processed_img, Default::default());
                            } else {
                                self.processed_image_texture = Some(
                                    ctx.load_texture("processed_image", egui_processed_img, Default::default())
                                );
                            }

                            self.status_message = format!(
                                "Effects applied with Bloom: S={:.1}, I={:.1}; Halation: S={:.1}, I={:.1}",
                                self.bloom_size, self.bloom_intensity, self.halation_size, self.halation_intensity
                            );
                        } else {
                            self.status_message = "Please load an image first.".to_string();
                        }
                    }
                    ui.label(&self.status_message);
                });


            // --- Image Display Area ---
            egui::CentralPanel::default().show_inside(ui, |ui| {
                ui.heading("Image Preview");
                ui.separator();
                if let Some(texture) = &self.processed_image_texture {
                    // Calculate aspect ratio to display image correctly
                    let available_width = ui.available_width();
                    let available_height = ui.available_height();
                    let tex_width = texture.size_vec2().x;
                    let tex_height = texture.size_vec2().y;

                    let aspect_ratio = tex_width / tex_height;
                    
                    let mut display_width = available_width;
                    let mut display_height = display_width / aspect_ratio;

                    if display_height > available_height {
                        display_height = available_height;
                        display_width = display_height * aspect_ratio;
                    }
                    
                    ui.image((texture.id(), egui::vec2(display_width, display_height)));

                } else {
                    ui.label("Load an image to see a preview.");
                    // Allocate space to prevent layout shift when image loads
                    ui.allocate_rect(ui.available_rect_before_wrap(), egui::Sense::hover());
                }
            });
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0]) // Increased initial window size
            .with_min_inner_size([640.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "BAH",
        options,
        Box::new(|_| {
            // You can customize egui visuals here if needed
            // For example, cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Box::<MyApp>::default()
        }),
    )
}
