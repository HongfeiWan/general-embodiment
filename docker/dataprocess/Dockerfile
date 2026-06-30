FROM isaac-gr00t-training-env:20260629

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_PREFIX=/opt/conda/envs/dataprocess
ENV PATH=/opt/conda/envs/dataprocess/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY system-libs/ /usr/local/lib/
RUN ldconfig

COPY dataprocess-conda-pack.tar.gz /tmp/dataprocess-conda-pack.tar.gz
RUN mkdir -p "$CONDA_PREFIX" \
    && tar -xzf /tmp/dataprocess-conda-pack.tar.gz -C "$CONDA_PREFIX" \
    && rm /tmp/dataprocess-conda-pack.tar.gz \
    && "$CONDA_PREFIX/bin/conda-unpack"

WORKDIR /workspace/general-embodiment
COPY entrypoint.sh /usr/local/bin/dataprocess-entrypoint
RUN chmod +x /usr/local/bin/dataprocess-entrypoint

EXPOSE 8501
ENTRYPOINT ["/usr/local/bin/dataprocess-entrypoint"]
CMD ["trim"]
