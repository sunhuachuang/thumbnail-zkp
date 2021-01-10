use image::{GenericImageView, ImageBuffer, Pixel};

use rand::prelude::*;
use std::time::{Duration, Instant};

use zkp_toolkit::bn_256::{Bn_256, Fr};
use zkp_toolkit::clinkv2::kzg10::{
    create_random_proof, verify_proof, ProveAssignment, VerifyAssignment, KZG10,
};
use zkp_toolkit::clinkv2::r1cs::{ConstraintSynthesizer, ConstraintSystem, SynthesisError};
use zkp_toolkit::math::{Field, One, PrimeField, Zero};

// Single round.
struct Thumbnail<F: PrimeField> {
    pub inps: [Option<F>; 100], // inputs pixel. 100-ratio
    pub out: Option<F>,         // outputs pixel.
    //pub p: Option<F>,           //position.
    pub p: u32,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for Thumbnail<F> {
    fn generate_constraints<CS: ConstraintSystem<F>>(
        self,
        cs: &mut CS,
        index: usize,
    ) -> Result<(), SynthesisError> {
        cs.alloc_input(|| "", || Ok(F::one()), index)?;

        //let var_p = cs.alloc_input(|| "y", || self.p.ok_or(SynthesisError::AssignmentMissing))?;
        let mut var_inps = vec![];
        for i in &self.inps {
            var_inps.push(cs.alloc(
                || "pre(inputs)",
                || i.ok_or(SynthesisError::AssignmentMissing),
                index,
            )?)
        }

        let var_o = cs.alloc_input(
            || "out(output)",
            || self.out.ok_or(SynthesisError::AssignmentMissing),
            index,
        )?;

        if index == 0 {
            cs.enforce(
                || "x * (y + 2) = z",
                |lc| lc + var_inps[self.p as usize],
                |lc| lc + CS::one(),
                |lc| lc + var_o,
            );
        }

        Ok(())
    }
}

fn as_bytes<T>(x: &T) -> &[u8] {
    use core::mem;
    use core::slice;

    unsafe { slice::from_raw_parts(x as *const T as *const u8, mem::size_of_val(x)) }
}

fn main() {
    //  Use the open function to load an image from a Path.
    // `open` returns a `DynamicImage` on success.
    let img = image::open("demo.jpg").unwrap();

    // The dimensions method returns the images width and height.
    let (x, y) = img.dimensions();
    println!("dimensions {:?}, {:?}", x, y);

    let n = 10u32; // ratio
    let p = 5u32; // position in matrix. 0 <= p <= n * n

    let new_x = x / n;
    let new_y = y / n;

    let mut rng = thread_rng();

    println!("Running mimc_clinkv2...");
    let m = new_x * new_y;

    // println!("Creating KZG10 parameters...");
    let degree = m.next_power_of_two() as usize;
    let mut crs_time = Duration::new(0, 0);

    // Create parameters for our circuit
    let start = Instant::now();

    let kzg10_pp = KZG10::<Bn_256>::setup(degree, false, &mut rng).unwrap();
    let (kzg10_ck, kzg10_vk) = KZG10::<Bn_256>::trim(&kzg10_pp, degree).unwrap();

    crs_time += start.elapsed();

    println!("Start prove prepare...");
    // Prover
    let prove_start = Instant::now();

    let mut prover_pa = ProveAssignment::<Bn_256>::default();
    let mut out_file = ImageBuffer::new(new_x, new_y);

    for m_x in 0..new_x {
        for m_y in 0..new_y {
            let mut tmp_pixels = [Some(Fr::zero()); 100];
            for i in 0..n {
                for j in 0..n {
                    let tmp_x = m_x * n + i;
                    let tmp_y = m_y * n + j;
                    let pixel = img.get_pixel(tmp_x, tmp_y).to_rgba();
                    let fr = Fr::from_random_bytes(as_bytes(&pixel)).unwrap();
                    if i * n + j == p {
                        out_file.put_pixel(m_x, m_y, pixel);
                    }
                    tmp_pixels[(i * n + j) as usize] = Some(fr);
                }
            }
            let tmp_out = tmp_pixels[p as usize].clone();

            let c = Thumbnail {
                inps: tmp_pixels,
                out: tmp_out,
                p: p,
            };
            c.generate_constraints(&mut prover_pa, (m_x * new_y + m_y) as usize)
                .unwrap();
        }
    }

    println!("Create prove...");
    // Create a clinkv2 proof with our parameters.
    let proof = create_random_proof(&prover_pa, &kzg10_ck, &mut rng).unwrap();
    let prove_time = prove_start.elapsed();

    out_file.save("test.png").unwrap();
    println!("Thumbnail image created: blocks: {}", new_x * new_y);

    // Verifier
    println!("Start verify prepare...");
    let verify_start = Instant::now();

    let mut verifier_pa = VerifyAssignment::<Bn_256>::default();

    // Create an instance of our circuit (with the witness)
    let verify_c = Thumbnail {
        inps: [None; 100],
        out: None,
        p: p,
    };
    verify_c
        .generate_constraints(&mut verifier_pa, 0usize)
        .unwrap();

    println!("Start verify...");

    let mut io: Vec<Vec<Fr>> = vec![];
    let mut output_fr = vec![];

    for m_x in 0..new_x {
        for m_y in 0..new_y {
            let pixel = out_file.get_pixel(m_x, m_y).to_rgba();
            let fr = Fr::from_random_bytes(as_bytes(&pixel)).unwrap();
            output_fr.push(fr);
        }
    }

    let one = vec![Fr::one(); (new_x * new_y) as usize];
    io.push(one);
    io.push(output_fr);

    // Check the proof
    assert!(verify_proof(&verifier_pa, &kzg10_vk, &proof, &io).unwrap());

    let verify_time = verify_start.elapsed();

    // Compute time

    let proving_avg =
        prove_time.subsec_nanos() as f64 / 1_000_000_000f64 + (prove_time.as_secs() as f64);
    let verifying_avg =
        verify_time.subsec_nanos() as f64 / 1_000_000_000f64 + (verify_time.as_secs() as f64);
    let crs_time = crs_time.subsec_nanos() as f64 / 1_000_000_000f64 + (crs_time.as_secs() as f64);

    println!("Generating CRS time: {:?}", crs_time);
    println!("Proving time: {:?}", proving_avg); // 48s
    println!("Verifying time: {:?}", verifying_avg); // 0.03s
}
