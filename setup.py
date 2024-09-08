from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
from pathlib import Path

class CustomBuildExt(build_ext):
    def run(self):
        # Run the original build_ext command to compile the C++ extension
        super().run()

        # Shader compilation
        shader_src = Path('cpp/add_tensor.comp')  # Path to your shader file
        shader_out = Path('cpp/add_tensor.spv')   # Output SPIR-V file
        
        if not shader_out.exists() or shader_src.stat().st_mtime > shader_out.stat().st_mtime:
            print(f"Compiling {shader_src} to {shader_out}")
            try:
                # Run the glslangValidator command to compile the .comp file to .spv
                subprocess.check_call([
                    'glslangValidator', '-V', str(shader_src), '-o', str(shader_out)
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
        )
    ]
)
