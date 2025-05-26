FROM python:3.12-slim-bookworm

RUN pip install uv supervisor --no-cache

WORKDIR /app
COPY . /app
RUN mv supervisord.conf /etc/supervisord.conf

RUN uv sync --no-cache && \
    pip cache purge && \
    uv cache clean

EXPOSE 8000

ENTRYPOINT ["supervisord"]
