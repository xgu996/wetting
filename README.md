You can try a demo by running the script at `src/test3_102_b90/main.py`.

For other examples, you might need to adjust the root directory manually. Make sure your working directory is set to `wetting/`. This can be configured in editors such as VSCode or Neovim. Alternatively, change your directory to `wetting/` and run:

```bash
PYTHONPATH=. python {relative_path_to_main.py}
```

The packages that have been successfully tested for the demo in `src/test3_102_b90/main.py` are listed in `requirements.txt`.

In this code, I used fealpy version 1.1.20. I have forked it in my repository [https://github.com/xgu996/fealpy](https://github.com/xgu996/fealpy).

Professor Wei has made significant improvements to fealpy in recent years. You can find his updated version at [https://github.com/weihuayi/fealpy](https://github.com/weihuayi/fealpy).

For some other demos, you might encounter issues with the provided version of netgen since I changed my PC a few years ago and no longer remember the exact version used. In such cases, please generate the mesh manually using netgen.

The code is designed to work with Python `3.8.18`.
