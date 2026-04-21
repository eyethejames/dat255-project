FROM node:22-bookworm-slim

ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHON_EXECUTABLE=/opt/venv/bin/python
ENV PORT=3000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
    && python3 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-deploy.txt

COPY webapp/package.json webapp/package-lock.json ./webapp/
RUN npm ci --prefix webapp --include=dev

COPY . .

RUN npm run build --prefix webapp \
    && /opt/venv/bin/python src/webapp_inference_runtime.py --warmup-only

EXPOSE 3000

CMD ["sh", "-c", "npm run start --prefix webapp -- -H 0.0.0.0 -p ${PORT:-3000}"]
