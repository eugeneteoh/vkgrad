from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
from pathlib import Path

class CustomBuildExt(build_ext):
    def run(self):
        # Run the original build_ext command to compile the C++ extension
        super().run()

        # Shader compilation
        shader_out = []
        shader_src = list(Path("cpp").glob("*.comp"))
        for path in shader_src:
            shader_out.append(path.with_suffix(".spv"))
        
        for src, out in zip(shader_src, shader_out):
            if not out.exists() or src.stat().st_mtime > out.stat().st_mtime:
                print(f"Compiling {src} to {out}")
                try:
                    # Run the glslangValidator command to compile the .comp file to .spv
                    subprocess.check_call([
                        'glslangValidator', '-V', str(src), '-o', str(out)
                    ])
                except subprocess.CalledProcessError as e:
                    print(f"Shader compilation failed: {e}")
                    raise


setup(
    name="vkgrad",
    packages=find_packages(),
    ext_modules=[
        Extension(
            name="vkgrad",
            sources=["cpp/tensor.cpp", "cpp/vulkan.cpp", "cpp/cpu.cpp"],
            language="c++",
            extra_compile_args=["-g", "-std=c++17"],
            extra_link_args=["-lvulkan"],
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExt,  # Override build_ext with custom class
    },
)
