use eerie::compiler;
use eerie::runtime;
use eerie::runtime::vm::ToRef;
use eerie::runtime::{
    hal::{BufferMapping, BufferView},
    vm::List,
};

use image::{DynamicImage, GenericImageView, Rgba, imageops::FilterType};
use ndarray::{Array4, ArrayBase};

fn image_to_yolo(original_image: &DynamicImage) -> Array4<f32> {
    let mut input = ArrayBase::zeros((1, 3, 640, 640));

    let image = original_image.resize_exact(640, 640, FilterType::CatmullRom);
    for (x, y, Rgba([r, g, b, _])) in image.pixels() {
        let x = x as usize;
        let y = y as usize;

        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    input
}

fn run(vmfb: &[u8], image_bin: &[f32]) -> Vec<f32> {
    let instance = runtime::api::Instance::new(
        &runtime::api::InstanceOptions::new(&mut runtime::hal::DriverRegistry::new())
            .use_all_available_drivers(),
    )
    .unwrap();

    let device = instance
        .try_create_default_device("local-task")
        .expect("Failed to create device");

    let session = runtime::api::Session::create_with_device(
        &instance,
        &runtime::api::SessionOptions::default(),
        &device,
    )
    .unwrap();

    unsafe { session.append_module_from_memory(vmfb) }.unwrap();

    let function = session.lookup_function("module.main_graph").unwrap();
    let input_list =
        runtime::vm::DynamicList::<runtime::vm::Ref<runtime::hal::BufferView<f32>>>::new(
            1, &instance,
        )
        .unwrap();

    let input_buffer = runtime::hal::BufferView::<f32>::new(
        &session,
        &[1, 640, 640, 3],
        runtime::hal::EncodingType::DenseRowMajor,
        image_bin,
    )
    .unwrap();

    let input_buffer_ref = input_buffer.to_ref(&instance).unwrap();
    input_list.push_ref(&input_buffer_ref).unwrap();

    let output_list =
        runtime::vm::DynamicList::<runtime::vm::Ref<runtime::hal::BufferView<f32>>>::new(
            1, &instance,
        )
        .unwrap();

    function.invoke(&input_list, &output_list).unwrap();

    let output_buffer_ref = output_list.get_ref(0).unwrap();
    let output_buffer: BufferView<f32> = output_buffer_ref.to_buffer_view(&session);
    let output_mapping = BufferMapping::new(output_buffer).unwrap();
    let out = output_mapping.data().to_vec();

    out
}

fn compile_mlir(data: &[u8]) -> Vec<u8> {
    let compiler = compiler::Compiler::new().unwrap();

    let targets = compiler.get_registered_hal_target_backends();
    println!("Registered HAL target backends: {:?}", targets);

    let mut compiler_session = compiler.create_session();

    compiler_session
        .set_flags(vec![
            "--iree-hal-target-backends=llvm-cpu".to_string(),
            "--iree-input-type=auto".to_string(),
        ])
        .unwrap();

    let source = compiler_session.create_source_from_buf(data).unwrap();
    let mut invocation = compiler_session.create_invocation();
    let mut output = compiler::MemBufferOutput::new(&compiler).unwrap();

    invocation
        .parse_source(source)
        .unwrap()
        .set_verify_ir(true)
        .set_compile_to_phase("end")
        .unwrap()
        .pipeline(compiler::Pipeline::Std)
        .unwrap()
        .output_vm_byte_code(&mut output)
        .unwrap();

    Vec::from(output.map_memory().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let start = std::time::Instant::now();

    // let mlir_bytecode = std::fs::read("yolo11x.mlir").unwrap();
    // let compiled_bytecode = compile_mlir(&mlir_bytecode);
    // std::fs::write("yolo11x_cpu_2.vmfb", &compiled_bytecode).unwrap();

    let compiled_bytecode = std::fs::read("yolo11x_cpu_2.vmfb").unwrap();

    println!("Compiled vmfb in {} ms", start.elapsed().as_millis());

    let start = std::time::Instant::now();

    let img = image::open("54c08a9-life-shura705.jpg")?;
    let yolo_im = image_to_yolo(&img);

    println!("Image to YOLO in {} ms", start.elapsed().as_millis());
    println!("Image: {:?}", yolo_im.shape());

    let image_bin = yolo_im.iter().map(|x| *x).collect::<Vec<f32>>();

    let output = run(&compiled_bytecode, &image_bin);

    println!("Run in {} ms", start.elapsed().as_millis());

    // let max_idx = output
    //     .iter()
    //     .enumerate()
    //     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //     .unwrap()
    //     .0;

    println!("{:#?}", output);

    // let id2label_file = std::fs::read_to_string("examples/id2label.txt").unwrap();

    // let id2label: Vec<&str> = id2label_file.split("\n").collect();

    // println!("The image is classified as: {}", id2label[max_idx]);

    Ok(())
}
