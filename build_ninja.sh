pip install ninja;
rm -rf /home/zeus/.cache/torch_extensions/py310_cu121;
export MAX_JOBS=12;

# should have read out like
# import os 

# os.environ['MAX_JOBS'] = "12"

# module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)