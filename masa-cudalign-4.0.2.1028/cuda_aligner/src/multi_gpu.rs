use std::sync::{Arc, Mutex};
use std::thread;
use crossbeam_channel::{bounded, Sender, Receiver};
use cuda_runtime_sys as cuda;

use crate::{CudaAlignerError, AlignmentResult, Sequence, AlignmentMode};

#[derive(Debug)]
pub struct GpuDevice {
    pub device_id: i32,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub name: String,
}

pub struct MultiGpuManager {
    devices: Vec<GpuDevice>,
    work_queues: Vec<(Sender<WorkItem>, Receiver<WorkResult>)>,
    workers: Vec<thread::JoinHandle<()>>,
}

struct WorkItem {
    seq1: Arc<Sequence>,
    seq2: Arc<Sequence>,
    mode: AlignmentMode,
}

struct WorkResult {
    device_id: i32,
    result: Result<AlignmentResult, CudaAlignerError>,
}

impl MultiGpuManager {
    pub fn new() -> Result<Self, CudaAlignerError> {
        let devices = Self::discover_devices()?;
        let mut work_queues = Vec::new();
        let mut workers = Vec::new();

        for device in &devices {
            let (work_tx, work_rx) = bounded(1);
            let (result_tx, result_rx) = bounded(1);
            
            let device = device.clone();
            let worker = thread::spawn(move || {
                Self::gpu_worker(device, work_rx, result_tx);
            });

            work_queues.push((work_tx, result_rx));
            workers.push(worker);
        }

        Ok(Self {
            devices,
            work_queues,
            workers,
        })
    }

    fn discover_devices() -> Result<Vec<GpuDevice>, CudaAlignerError> {
        let mut devices = Vec::new();
        unsafe {
            let mut device_count = 0;
            cuda::cudaGetDeviceCount(&mut device_count);

            for device_id in 0..device_count {
                let mut props = std::mem::zeroed::<cuda::cudaDeviceProp>();
                cuda::cudaGetDeviceProperties(&mut props, device_id);

                devices.push(GpuDevice {
                    device_id,
                    compute_capability: (props.major, props.minor),
                    total_memory: props.totalGlobalMem as usize,
                    name: String::from_utf8_lossy(&props.name[..])
                        .trim_matches(char::from(0))
                        .to_string(),
                });
            }
        }
        Ok(devices)
    }

    fn gpu_worker(
        device: GpuDevice,
        work_rx: Receiver<WorkItem>,
        result_tx: Sender<WorkResult>,
    ) {
        while let Ok(work) = work_rx.recv() {
            // Set GPU device for this thread
            unsafe {
                cuda::cudaSetDevice(device.device_id);
            }

            // Create aligner for this GPU
            let mut aligner = match CudaAligner::new(
                CudaAlignerParameters::new()
                    .with_gpu(device.device_id)
                    .with_blocks(256)
                    .expect("Invalid block count")
            ) {
                Ok(aligner) => aligner,
                Err(e) => {
                    let _ = result_tx.send(WorkResult {
                        device_id: device.device_id,
                        result: Err(e),
                    });
                    continue;
                }
            };

            // Perform alignment
            let result = aligner
                .with_alignment_mode(work.mode)
                .align_with_traceback();

            let _ = result_tx.send(WorkResult {
                device_id: device.device_id,
                result,
            });
        }
    }

    pub fn align_parallel(
        &self,
        sequences: Vec<(Sequence, Sequence)>,
        mode: AlignmentMode,
    ) -> Result<Vec<AlignmentResult>, CudaAlignerError> {
        let mut results = Vec::with_capacity(sequences.len());
        let mut work_items = sequences.into_iter().enumerate();
        let mut active_work = 0;

        // Initial distribution of work
        for (tx, _) in &self.work_queues {
            if let Some((_, (seq1, seq2))) = work_items.next() {
                tx.send(WorkItem {
                    seq1: Arc::new(seq1),
                    seq2: Arc::new(seq2),
                    mode,
                }).map_err(|e| CudaAlignerError::MultiGpuError(e.to_string()))?;
                active_work += 1;
            }
        }

        // Process results and distribute remaining work
        while active_work > 0 {
            for (work_tx, result_rx) in &self.work_queues {
                if let Ok(work_result) = result_rx.try_recv() {
                    active_work -= 1;
                    
                    // Process result
                    match work_result.result {
                        Ok(result) => results.push(result),
                        Err(e) => return Err(e),
                    }

                    // Distribute more work if available
                    if let Some((_, (seq1, seq2))) = work_items.next() {
                        work_tx.send(WorkItem {
                            seq1: Arc::new(seq1),
                            seq2: Arc::new(seq2),
                            mode,
                        }).map_err(|e| CudaAlignerError::MultiGpuError(e.to_string()))?;
                        active_work += 1;
                    }
                }
            }
            thread::yield_now();
        }

        Ok(results)
    }
}

impl Drop for MultiGpuManager {
    fn drop(&mut self) {
        // Close work channels to stop workers
        self.work_queues.clear();

        // Wait for workers to finish
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
} 