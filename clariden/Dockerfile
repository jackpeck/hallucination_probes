FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace

# Copy your project
RUN git clone https://github.com/sevdari/hallucination_probes.git
WORKDIR /workspace/hallucination_probes

# Make sure pip is recent but DO NOT install torch
RUN python -m pip install --upgrade pip

# Install your package into system Python without dependencies
# (torch from NVIDIA image stays untouched)
RUN python -m pip install  .


CMD ["/bin/bash"]
