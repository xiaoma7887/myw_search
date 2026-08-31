@echo off
call D:\dev\anaconda\Scripts\activate.bat text2img
python -c "import sys; print('Python:', sys.executable)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
pause

