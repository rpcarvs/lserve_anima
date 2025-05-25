FROM python:3.12-slim-bookworm

####### Add your own installation commands here #######
RUN pip install uv --no-cache
RUN apt update && apt install -y \
    supervisor 

WORKDIR /app
COPY . /app
RUN mv supervisord.conf /etc/supervisor/supervisord.conf

RUN uv sync --no-cache && \
    pip cache purge && \
    uv cache clean && \

EXPOSE 8000
ENTRYPOINT ["supervisord"]
