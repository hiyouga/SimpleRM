1. Ensure that you have installed `torch>=2.1.0` and `flash_attn`.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the training script using `bash train.sh optim.global_batch_size=128`.
4. Run the evaluation script using `bash eval.sh optim.micro_batch_size=32`.
5. Run the inference script using `python inference.py model_path=output`.
