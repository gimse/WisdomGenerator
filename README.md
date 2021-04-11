# WisdomGenerator
The goal is to recreate them model made in this [article](https://towardsdatascience.com/how-to-train-and-deploy-custom-ai-generated-quotes-using-gpt2-fastapi-and-reactjs-9a6feb42d8b0)
That is, the goal is the make a Norwegian version of [inspirobot](https://www.instagram.com/inspirobot.me/?hl=en).

## Setup (for Anaconda)
- Install [Anaconda](https://www.anaconda.com)
- Make virtuale envirorment: ``conda create -n wisdomgenerator python=3.8.8 anaconda``
- ``conda activate wisdomgenerator``
- ``conda install -r requirements.txt``
- Check that the pretrained model is working by running [test_pretrained_model.py](test_pretrained_model.py).
- Check that the GPU is working by running [check_if_using_gpu.py](check_if_using_gpu.py)
- Start training on a very small data set, to se if it is working 