# 可复现运行环境：docker build -t emslice . && docker run --rm -v $PWD/data:/app/data emslice emslice info
FROM condaforge/miniforge3:24.11.3-0
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN mamba env create -f environment.yml && mamba clean -afy
SHELL ["conda", "run", "-n", "emslice", "/bin/bash", "-c"]
COPY . /app
RUN pip install --no-deps -e . && python -m pytest -q
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "emslice"]
CMD ["emslice", "info"]
