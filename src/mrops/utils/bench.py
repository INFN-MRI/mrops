"""Case generation for testing, examples and benchmarks."""

import mrtwin
import torchsim

import sigpy

def generate_simple_data():
    ...
    
def generate_stack_data(mode: str = "static"):
    obj = mrtwin.brainweb_phantom(
        ndim=3, 
        subject=4, 
        segtype=False, 
        shape=(144, 220, 220), 
        output_res=1.0
        )
    
def generate_isotropic_data():
    ...
    